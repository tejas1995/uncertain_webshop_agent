import json
import yaml
import pdb
import os
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import time
import random
import re
from pprint import pformat
import argparse
import string
from tqdm import tqdm

from agent import Agent
from webshop_env import WebShopEnv


# First, set root logger to WARNING to suppress other modules
logging.getLogger().setLevel(logging.WARNING)

# Remove the basicConfig since we want to control logging more precisely
logger = logging.getLogger(__name__)

# Create a console handler and set its level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to our logger
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

def run_rollout(agent, env, args):
    trajectory = []
    observation_instruction_text = env.get_instruction_text().replace("Instruction:", "Instruction:\n")
    trajectory.append({"role": "user", "content": "USER COMMAND: " + observation_instruction_text.replace("Instruction:\n", '').strip()})
    env_observation = env.get_observation().replace(observation_instruction_text, "")
    trajectory.append({"role": "user", "content": f"ENVIRONMENT OBSERVATION: {env_observation}"})

    num_environment_steps = 0
    num_errors = 0
    trajectory_start_time = time.time()
    done = False
    while not done and num_environment_steps < args.max_steps:
        try:
            agent_output = agent.get_action(trajectory)
            assert "ACTION:" in agent_output, "Agent response must contain an action."
            action = agent_output.split("ACTION:")[-1].replace("\n", "").strip()
            trajectory.append({"role": "assistant", "content": agent_output})
            # logger.info(f"Action #{num_environment_steps}: {action}")

            obs, reward, done, info = env.step(action)
            obs = obs.replace(observation_instruction_text, "")
            trajectory.append({"role": "user", "content": f"ENVIRONMENT OBSERVATION: {obs}"})
            num_environment_steps += 1
        except Exception as e:
            num_errors += 1
            if num_errors >= 10:
                return [], 0, {"num_environment_steps": 0, "llm_cost": 0, "trajectory_time": 0, "num_errors": num_errors}
            continue

    if info is None:
        info = {}
    info["num_environment_steps"] = num_environment_steps
    info["llm_cost"] = agent.llm_cost
    info["trajectory_time"] = time.time() - trajectory_start_time
    info["num_errors"] = num_errors
    info["task_completed"] = done
    return trajectory, reward, info

def run_rollouts(agent, env, args):
    rollouts = []
    for i in tqdm(range(args.num_tasks)):
        env.initialize_environment(i)
        agent.reset()
        rollout = run_rollout(agent, env, args)
        trajectory, reward, info = rollout
        rollouts.append({
            "goal_idx": i,
            "goal": env.env.server.goals[i],
            "trajectory": trajectory,
            "reward": reward,
            "info": info
        })
        logger.info(f"Task {i}: reward={reward:.4f}, num_errors={info['num_errors']}")

    return rollouts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_config_file", type=str, default="agent_config.yaml")
    parser.add_argument("--secrets_file", type=str, default="../secrets.yaml")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--num_tasks", type=int, default=100)
    parser.add_argument("--rollout_save_dir", type=str, default="experiments/noninteractive_rollouts")
    parser.add_argument("--experiment_name", type=str,  required=True, default="noninteractive_rollouts")
    args = parser.parse_args()

    os.makedirs(args.rollout_save_dir, exist_ok=True)
    args.experiment_name = f"{args.experiment_name}-{args.num_tasks}tasks-{args.max_steps}maxsteps"

    with open(args.agent_config_file, "r") as f:
        agent_config = yaml.safe_load(f)

    agent = Agent(agent_config, args)
    env = WebShopEnv(args)
    rollouts = run_rollouts(agent, env, args)

    avg_reward = np.mean([rollout["reward"] for rollout in rollouts])
    total_llm_cost = np.sum([rollout["info"]["llm_cost"] for rollout in rollouts])
    total_time = np.sum([rollout["info"]["trajectory_time"] for rollout in rollouts])
    logger.info(f"Average reward: {avg_reward:.4f}")
    logger.info(f"Total LLM cost: ${total_llm_cost:.2f}")
    logger.info(f"Total time: {total_time:.2f}s")

    results_data = {
        "avg_reward": avg_reward,
        "num_perfect_rollouts": sum([rollout["reward"] == 1.0 for rollout in rollouts]),
        "num_total_failure_rollouts": sum([rollout["reward"] == 0.0 for rollout in rollouts]),
        "num_task_completed_rollouts": sum([rollout["info"]["task_completed"] for rollout in rollouts]),
        "llm_cost": total_llm_cost,
        "total_time": total_time,
        "rollouts": rollouts,
    }
    with open(os.path.join(args.rollout_save_dir, f"{args.experiment_name}.json"), "w") as f:
        json.dump(results_data, f, indent=4)
    logger.info(f"Results saved to {os.path.join(args.rollout_save_dir, f'{args.experiment_name}.json')}")