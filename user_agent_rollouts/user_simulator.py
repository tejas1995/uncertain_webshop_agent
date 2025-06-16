import yaml
import pdb
from typing import List, Dict
import anthropic

cost_dict = {
    "claude-3-7-sonnet-latest": {"input": 3/10**6, "output": 15/10**6},
}

class UserSimulator:

    def __init__(self, user_config: dict, args: dict):
        self.user_config = user_config
        self.args = args
        secrets = yaml.safe_load(open(args.secrets_file, "r"))
        self.anthropic_client = anthropic.Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
        self.llm_cost = 0
        self.scenario = None
        self.trajectory = []
        self.system_prompt = self.user_config["initial_system_prompt"]
        self.current_system_prompt = self.system_prompt

    def get_user_utterance(self, agent_message: str = None) -> str:
        """
        Creates input messages based on system prompts and session trajectory, then queries the user for a response.
        """
        input_messages = []
        for msg in self.trajectory:
            if isinstance(msg, str):
                input_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                input_messages.append({"role": msg["role"], "content": msg["content"]})
        if agent_message is not None:
            input_messages.append({"role": "user", "content": f"ASSISTANT RESPONSE: {agent_message}"})
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