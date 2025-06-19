import yaml
import pdb
from typing import List, Dict
import anthropic
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import logging

# First, set root logger to WARNING to suppress other modules
logging.getLogger().setLevel(logging.WARNING)

# Remove the basicConfig since we want to control logging more precisely
logger = logging.getLogger(__name__)

cost_dict = {
    "claude-3-7-sonnet-latest": {"input": 3/10**6, "output": 15/10**6},
}

class Agent:

    def __init__(self, agent_config: dict, args: dict):
        self.agent_config = agent_config
        self.args = args
        self.secrets = yaml.safe_load(open(args.secrets_file, "r"))
        self.llm_cost = 0
        self.input_tokens = []


    def get_action(self, trajectory: List[Dict]) -> str:
        pass

    def reset(self):
        self.llm_cost = 0

class ClaudeAgent(Agent):
    def __init__(self, agent_config: dict, args: dict):
        super().__init__(agent_config, args)
        self.anthropic_client = anthropic.Anthropic(api_key=self.secrets["ANTHROPIC_API_KEY"])
        assert self.agent_config["model_name"] in cost_dict, f"Model {self.agent_config['model_name']} not found in cost_dict"
        logger.info(f"Initialized ClaudeAgent with model: {self.agent_config['model_name']}")


    def get_action(self, trajectory: List[Dict]) -> str:
        """
        Creates input messages based on system prompts and session trajectory,
        then queries the agent for an action
        """
        input_messages = []
        # input_messages.append({"role": "system", "content": self.agent_config["initial_system_prompt"]})
        
        # Find the last occurrence of 'ENVIRONMENT OBSERVATION:'
        last_env_obs_index = -1
        for i, msg in enumerate(trajectory):
            if isinstance(msg, dict) and "content" in msg and "ENVIRONMENT OBSERVATION:" in msg["content"]:
                last_env_obs_index = i
        
        # Add messages to input_messages, filtering out environment observations except the last one
        for i, msg in enumerate(trajectory):
            if isinstance(msg, str):
                input_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                # Skip environment observations unless it's the last one
                if self.agent_config["env_obs_history"] == "last" and "ENVIRONMENT OBSERVATION:" in msg["content"] and i != last_env_obs_index:
                    continue
                input_messages.append({"role": msg["role"], "content": msg["content"]})
        
        input_messages.append({"role": "user", "content": self.agent_config["final_system_prompt"]})

        agent_response = self.anthropic_client.messages.create(
            model=self.agent_config["model_name"],
            system=self.agent_config["initial_system_prompt"],
            messages=input_messages,
            max_tokens=self.agent_config["max_tokens"],
            temperature=self.agent_config["temperature"],
        )
        input_tokens = agent_response.usage.input_tokens
        output_tokens = agent_response.usage.output_tokens
        cost = cost_dict[self.agent_config["model_name"]]["input"] * input_tokens + \
               cost_dict[self.agent_config["model_name"]]["output"] * output_tokens
        self.llm_cost += cost
        self.input_tokens.append(input_tokens)

        return agent_response.content[0].text

class LlamaAgent(Agent):
    def __init__(self, agent_config: dict, args: dict):
        super().__init__(agent_config, args)
        self.model = LlamaForCausalLM.from_pretrained(
            self.agent_config["model_name"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.agent_config["model_name"])
        logger.info(f"Initialized LlamaAgent with model: {self.agent_config['model_name']}")

    def get_action(self, trajectory: List[Dict]) -> str:
        """
        Creates input messages based on system prompts and session trajectory,
        then queries the agent for an action
        """
        input_messages = []
        if self.agent_config["initial_system_prompt"] is not None:
            input_messages.append({"role": "system", "content": self.agent_config["initial_system_prompt"]})
        
        # Find the last occurrence of 'ENVIRONMENT OBSERVATION:'
        last_env_obs_index = -1
        for i, msg in enumerate(trajectory):
            if isinstance(msg, dict) and "content" in msg and "ENVIRONMENT OBSERVATION:" in msg["content"]:
                last_env_obs_index = i
        
        # Add messages to input_messages, filtering out environment observations except the last one
        for i, msg in enumerate(trajectory):
            if isinstance(msg, str):
                input_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                # Skip environment observations unless it's the last one
                if self.agent_config["env_obs_history"] == "last" and "ENVIRONMENT OBSERVATION:" in msg["content"] and i != last_env_obs_index:
                    continue
                input_messages.append({"role": msg["role"], "content": msg["content"]})

        if self.agent_config["final_system_prompt"] is not None:
            input_messages.append({"role": "system", "content": self.agent_config["final_system_prompt"]})

        input_ids = self.tokenizer.apply_chat_template(
            input_messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=self.agent_config["max_tokens"],
            temperature=self.agent_config["temperature"],
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_text = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        input_tokens = input_ids.shape[1]
        self.input_tokens.append(input_tokens)
        output_text = output_text.strip('\n').strip()

        # print("INPUT:", self.tokenizer.decode(input_ids[0]))
        # print("OUTPUT:", output_text)
        # pdb.set_trace()
        return output_text

model_type_to_class = {
    "claude": ClaudeAgent,
    "llama": LlamaAgent,
}

def get_agent(agent_config: dict, args: dict):
    model_type = agent_config["model_type"]
    assert model_type in model_type_to_class, f"Model type {model_type} not found in model_type_to_class"
    return model_type_to_class[model_type](agent_config, args)