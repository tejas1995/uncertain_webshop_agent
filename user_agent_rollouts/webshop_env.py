import json
import numpy as np
import random
import time
from minimal_webshop.envs.web_agent_text_env import WebAgentTextEnv

class WebShopEnv:

    def __init__(self, args: dict):
        self.args = args
        self.env = WebAgentTextEnv(
            observation_mode="text_rich",
            human_goals=True,
        )
        self.scenarios_loaded = False

    def reset(self, scenario_index: int):
        return self.env.reset(scenario_index)
    
    def step(self, action: str):
        return self.env.step(action)
    
    def load_env(self):
        if self.env is None:
            self.env = WebAgentTextEnv(
                observation_mode="text_rich",
                human_goals=True,
            )

    def load_scenarios(self):
        with open(self.args.scenarios_file, "r") as f:
            scenarios = json.load(f)
            # random.seed(time.time())
            random.seed(233)
            random.shuffle(scenarios)
        self.env.server.goals = scenarios
        self.env.server.weights = [goal['weight'] for goal in scenarios]
        self.env.server.cum_weights = [0] + np.cumsum(self.env.server.weights).tolist()


    def initialize_environment(self, scenario_index):
        self.load_env()
        if "scenarios_file" in self.args and self.args.scenarios_file is not None and not self.scenarios_loaded:
            self.load_scenarios()
            self.scenarios_loaded = True
        self.env.reset(scenario_index)

    def get_instruction_text(self):
        return self.env.instruction_text
    
    def get_observation(self):
        return self.env.observation