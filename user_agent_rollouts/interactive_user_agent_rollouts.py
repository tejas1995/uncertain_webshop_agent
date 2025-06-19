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

from webshop_env import WebShopEnv
from agent import get_agent
from user_simulator import get_user_simulator
from eval_utils import WebShopLMEvaluator

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

def run_rollout(agent, env, user_simulator, evaluator, args):
    trajectory = []
    initial_user_message = user_simulator.get_user_utterance()
    trajectory.append({"role": "user", "content": f"USER INITIAL INSTRUCTION: {initial_user_message}"})
    logger.info(f"Initial user message: {initial_user_message}")
    reward = 0.0
    info = None

    observation_instruction_text = env.get_instruction_text().replace("Instruction:", "Instruction:\n")

    num_environment_steps = 0
    num_errors = 0
    trajectory_start_time = time.time()
    done = False
    while not done and num_environment_steps < args.max_steps:
        try:
            agent_output = agent.get_action(trajectory)
            assert "ACTION:" in agent_output, f"Agent response must contain an action. Agent output: {agent_output}"
            action = agent_output.split("ACTION:")[-1].replace("\n", "").strip()
            trajectory.append({"role": "assistant", "content": agent_output})
            logger.info(f"Action #{num_environment_steps}: {action}")

            if action.startswith("respond["):
                agent_message = action.split("respond[")[-1].strip("]").strip()
                user_response = user_simulator.get_user_utterance(agent_message)
                trajectory.append({"role": "user", "content": f"USER RESPONSE: {user_response}"})
                logger.info(f"User response: {user_response}")
                # pdb.set_trace()
                
            else:
                obs, reward, done, info = env.step(action)
                obs = obs.replace(observation_instruction_text, "")
                trajectory.append({"role": "user", "content": f"ENVIRONMENT OBSERVATION: {obs}"})
            num_environment_steps += 1
        except Exception as e:
            num_errors += 1
            print(str(e))
            # import pdb; pdb.set_trace()
            if num_errors >= 10:
                return trajectory, 0, {
                    "num_environment_steps": 0, 
                    "llm_cost": 0, 
                    "trajectory_time": 0, 
                    "num_errors": num_errors, 
                    "evaluator_reward": 0.0, 
                    "task_completed": False,
                    "agent_llm_cost": agent.llm_cost, 
                    "user_llm_cost": user_simulator.llm_cost, 
                    "total_llm_cost": agent.llm_cost + user_simulator.llm_cost,
                    "num_user_utterances": len([x for x in user_simulator.trajectory if x["role"] == "assistant"]),
                    "purchased_item": None,
                }
            continue

    if info is None:
        info = {}
    info["num_environment_steps"] = num_environment_steps
    info["agent_llm_cost"] = agent.llm_cost
    info["user_llm_cost"] = user_simulator.llm_cost
    info["total_llm_cost"] = agent.llm_cost + user_simulator.llm_cost
    info["num_user_utterances"] = len([x for x in user_simulator.trajectory if x["role"] == "assistant"])
    info["trajectory_time"] = time.time() - trajectory_start_time
    info["num_errors"] = num_errors
    info["task_completed"] = done
    if done and info["purchased_item"] is not None:
        evaluator_results = evaluator.evaluate(info["purchased_item"], user_simulator.scenario)
        info["evaluator_results"] = evaluator_results
        info["evaluator_reward"] = evaluator_results["total_reward"]
    else:
        info["evaluator_reward"] = 0.0
    return trajectory, reward, info

def run_rollouts(agent, env, user_simulator, evaluator, args):
    rollouts = []
    for i in tqdm(range(args.num_tasks)):
        env.initialize_environment(i)
        agent.reset()
        user_simulator.reset(env.env.server.goals[i])
        rollout = run_rollout(agent, env, user_simulator, evaluator, args)
        trajectory, reward, info = rollout
        rollouts.append({
            "scenario_idx": i,
            "scenario": env.env.server.goals[i],
            "trajectory": trajectory,
            "reward": reward,
            "evaluator_reward": info["evaluator_reward"],
            "info": info
        })
        logger.info(f"Task {i}: reward={reward:.4f}, num_errors={info['num_errors']}")

    return rollouts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_config_file", type=str, default="agent_config.yaml")
    parser.add_argument("--user_config_file", type=str, default="user_config.yaml")
    parser.add_argument("--scenarios_file", type=str, required=True)
    parser.add_argument("--secrets_file", type=str, default="../secrets.yaml")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--num_tasks", type=int, default=100)
    parser.add_argument("--evaluator_model_name", type=str, default="claude-3-7-sonnet-latest")
    parser.add_argument("--rollout_save_dir", type=str, default="/data/env/lib/repos/tejas_srinivasan/webshop_experiments/interactive_rollouts")
    parser.add_argument("--experiment_name", type=str,  required=True)
    args = parser.parse_args()

    # Load environment, agent, user simulator, and evaluator
    env = WebShopEnv(args)
    env.load_scenarios()
    with open(args.agent_config_file, "r") as f:
        agent_config = yaml.safe_load(f)
    agent = get_agent(agent_config, args)
    with open(args.user_config_file, "r") as f:
        user_config = yaml.safe_load(f)
    user_simulator = get_user_simulator(user_config, args)
    evaluator = WebShopLMEvaluator(args.evaluator_model_name, args)

    if args.num_tasks > len(env.env.server.goals):
        logger.warning(f"Number of tasks ({args.num_tasks}) is greater than number of scenarios ({len(env.env.server.goals)}). Setting num_tasks to {len(env.env.server.goals)}.")
        args.num_tasks = len(env.env.server.goals)

    rollouts = run_rollouts(agent, env, user_simulator, evaluator, args)

    avg_reward = np.mean([rollout["reward"] for rollout in rollouts]).astype(float)
    avg_evaluator_reward = np.mean([rollout["evaluator_reward"] for rollout in rollouts]).astype(float)
    total_llm_cost = np.sum([rollout["info"]["total_llm_cost"] for rollout in rollouts]).astype(float)
    total_time = np.sum([rollout["info"]["trajectory_time"] for rollout in rollouts]).astype(float)
    avg_num_user_utterances = sum([rollout["info"]["num_user_utterances"] for rollout in rollouts]) / len(rollouts)
    logger.info(f"Average reward: {avg_reward:.4f}")
    logger.info(f"Average evaluator reward: {avg_evaluator_reward:.4f}")
    logger.info(f"Total (user + agent + evaluator) LLM cost: ${total_llm_cost + evaluator.llm_cost:.2f}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average number of user utterances: {avg_num_user_utterances:.2f}")

    results_data = {
        "avg_reward": avg_reward,
        "avg_evaluator_reward": avg_evaluator_reward,
        "num_perfect_rollouts": sum([rollout["reward"] == 1.0 for rollout in rollouts]),
        "num_total_failure_rollouts": sum([rollout["reward"] == 0.0 for rollout in rollouts]),
        "num_task_completed_rollouts": sum([rollout["info"]["task_completed"] for rollout in rollouts]),
        "avg_num_user_utterance": avg_num_user_utterances,
        "agent_llm_cost": sum([rollout["info"]["agent_llm_cost"] for rollout in rollouts]),
        "user_llm_cost": sum([rollout["info"]["user_llm_cost"] for rollout in rollouts]),
        "evaluator_llm_cost": evaluator.llm_cost,
        "total_llm_cost": total_llm_cost + evaluator.llm_cost,
        "total_time": total_time,
        "rollouts": rollouts,
    }

    if args.num_tasks != len(env.env.server.goals):
        args.experiment_name = f"{args.experiment_name}-{args.num_tasks}tasks-{args.max_steps}maxsteps"
    else:
        args.experiment_name = f"{args.experiment_name}-{args.max_steps}maxsteps"
    out_filename = os.path.join(args.rollout_save_dir, f"{args.experiment_name}.json")
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    try:
        with open(out_filename, "w") as f:
            json.dump(results_data, f, indent=4)
        logger.info(f"Results saved to {out_filename}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        pdb.set_trace()

    pdb.set_trace()