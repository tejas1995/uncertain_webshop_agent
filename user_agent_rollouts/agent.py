import yaml
from typing import List, Dict
import anthropic

cost_dict = {
    "claude-3-7-sonnet-latest": {"input": 3/10**6, "output": 15/10**6},
}

class Agent:

    def __init__(self, agent_config: dict, args: dict):
        self.agent_config = agent_config
        self.args = args
        secrets = yaml.safe_load(open(args.secrets_file, "r"))
        self.anthropic_client = anthropic.Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
        self.llm_cost = 0
        self.input_tokens = []

        assert self.agent_config["model_name"] in cost_dict, f"Model {self.agent_config['model_name']} not found in cost_dict"

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

    def reset(self):
        self.llm_cost = 0
