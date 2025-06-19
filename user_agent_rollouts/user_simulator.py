import yaml
import pdb
from typing import List, Dict
import anthropic
import logging
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

cost_dict = {
    "claude-3-7-sonnet-latest": {"input": 3/10**6, "output": 15/10**6},
}

class UserSimulator:

    def __init__(self, user_config: dict, args: dict):
        self.user_config = user_config
        self.args = args
        self.secrets = yaml.safe_load(open(args.secrets_file, "r"))
        self.llm_cost = 0
        self.scenario = None
        self.trajectory = []
        self.system_prompt = self.user_config["initial_system_prompt"]
        self.current_system_prompt = self.system_prompt
        self.input_tokens = []

    def get_user_utterance(self, agent_message: str = None) -> str:
        pass

    def reset(self, scenario: dict):
        self.scenario = scenario
        self.trajectory = []
        self.current_system_prompt = self.system_prompt.format(
            scenario=scenario['instruction_text'],
            product=scenario['product_type'],
            # attributes=', '.join(scenario['attributes']),
            attributes="You want the product to have the following attributes:\n" + scenario['attribute_wise_reasoning'],
            preferences='\n'.join([f"- {k}: {v}" for k, v in scenario['goal_options_dict'].items()]),
            budget=f"${scenario['price_upper']:.0f}"
        )

class ClaudeUserSimulator(UserSimulator):
    def __init__(self, user_config: dict, args: dict):
        super().__init__(user_config, args)
        self.anthropic_client = anthropic.Anthropic(api_key=self.secrets["ANTHROPIC_API_KEY"])
        assert self.user_config["model_name"] in cost_dict, f"Model {self.user_config['model_name']} not found in cost_dict"
        logger.info(f"Initialized ClaudeUserSimulator with model: {self.user_config['model_name']}")

    def get_user_utterance(self, agent_message: str = None) -> str:
        """
        Creates input messages based on system prompts and session trajectory, then queries the user for a response.
        """
        if agent_message is not None:
            self.trajectory.append({"role": "user", "content": f"ASSISTANT RESPONSE: {agent_message}"})

        input_messages = []
        for msg in self.trajectory:
            if isinstance(msg, str):
                input_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                input_messages.append({"role": msg["role"], "content": msg["content"]})
        input_messages.append({"role": "user", "content": self.user_config["final_system_prompt"]})

        user_response = self.anthropic_client.messages.create(
            model=self.user_config["model_name"],
            system=self.current_system_prompt,
            messages=input_messages,
            max_tokens=self.user_config["max_tokens"],
            temperature=self.user_config["temperature"],
        )
        input_tokens = user_response.usage.input_tokens
        output_tokens = user_response.usage.output_tokens
        cost = cost_dict[self.user_config["model_name"]]["input"] * input_tokens + \
               cost_dict[self.user_config["model_name"]]["output"] * output_tokens
        self.llm_cost += cost
        self.trajectory.append({"role": "assistant", "content": user_response.content[0].text})
        return user_response.content[0].text

class LlamaUserSimulator(UserSimulator):
    def __init__(self, user_config: dict, args: dict):
        super().__init__(user_config, args)
        self.model = LlamaForCausalLM.from_pretrained(
            self.user_config["model_name"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.user_config["model_name"])
        logger.info(f"Initialized LlamaAgent with model: {self.user_config['model_name']}")

    def get_user_utterance(self, agent_message: str = None) -> str:
        """
        Creates input messages based on system prompts and session trajectory, then queries the user for a response.
        """
        if agent_message is not None:
            self.trajectory.append({"role": "user", "content": f"ASSISTANT RESPONSE: {agent_message}"})

        input_messages = [{"role": "system", "content": self.current_system_prompt}]
        for msg in self.trajectory:
            if isinstance(msg, str):
                input_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                input_messages.append({"role": msg["role"], "content": msg["content"]})
        input_messages.append({"role": "system", "content": self.user_config["final_system_prompt"]})

        input_ids = self.tokenizer.apply_chat_template(
            input_messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=self.user_config["max_tokens"],
            temperature=self.user_config["temperature"],
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
        self.trajectory.append({"role": "assistant", "content": output_text})

        # print("INPUT:", self.tokenizer.decode(input_ids[0]))
        # print("OUTPUT:", output_text)
        # pdb.set_trace()
        return output_text

model_type_to_class = {
    "claude": ClaudeUserSimulator,
    "llama": LlamaUserSimulator,
}

def get_user_simulator(user_config: dict, args: dict):
    model_type = user_config["model_type"]
    assert model_type in model_type_to_class, f"Model type {model_type} not found in model_type_to_class"
    return model_type_to_class[model_type](user_config, args)