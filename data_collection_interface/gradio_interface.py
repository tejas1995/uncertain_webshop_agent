import gradio as gr
import json
import yaml
import traceback
import copy
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

from minimal_webshop.envs.web_agent_text_env import WebAgentTextEnv

# Set root logger to WARNING to disable most logs
logging.getLogger().setLevel(logging.WARNING)

# Configure only the main module logger to show INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scenarios_file", 
    help="Path to the scenarios file"
)
parser.add_argument(
    "--agent_config_file", 
    default="agent_config.yaml",
    help="Path to the agent config file"
)
parser.add_argument(
    "--secrets_file", 
    default="../secrets.yaml",
    help="Path to the secrets file"
)
parser.add_argument(
    "--save_dir",
    default="session_data",
    help="Path to the save directory"
)
args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)

# Create global environment and keep reference to its server
env = WebAgentTextEnv(observation_mode="text_rich", human_goals=True)
with open(args.scenarios_file, "r") as f:
    scenarios = json.load(f)
env.server.goals = scenarios
env.server.weights = [goal['weight'] for goal in scenarios]
env.server.cum_weights = [0] + np.cumsum(env.server.weights).tolist()
# Keep reference to the server with initialized search engine etc.
global_server = env.server
logger.info(f"Global server initialized with {len(global_server.goals)} goals")

def show_satisfaction_form(trajectory_id):
    """Show the satisfaction form and store the trajectory_id"""
    return {
        "is_visible": True,
        "trajectory_id": trajectory_id
    }

def submit_satisfaction_form(satisfied: str, conforms_to_scenario: str, assistant_helpful: str, annotator_name: str, session_id: str, form_state: dict):
    """Save satisfaction form data and hide the form"""
    if not form_state["is_visible"] or not form_state["trajectory_id"]:
        return {
            "is_visible": False,
            "trajectory_id": None
        }, "No active form to submit."

    # Save feedback to the session file
    filename = f"{args.save_dir}/annotator_{annotator_name.lower().replace(' ', '_')}-session_{session_id}.json"
    try:
        with open(filename, "r") as f:
            session_data = json.load(f)
        
        # Add feedback to the trajectory
        if form_state["trajectory_id"] in session_data["trajectories"]:
            session_data["trajectories"][form_state["trajectory_id"]]["user_feedback"] = {
                "product_feedback": satisfied,
                "scenario_conformance": conforms_to_scenario,
                "assistant_helpfulness": assistant_helpful
            }
            
            with open(filename, "w") as f:
                json.dump(session_data, f, indent=4)
            
            return {
                "is_visible": False,
                "trajectory_id": None
            }, "✅ Feedback saved successfully."
    except Exception as e:
        return form_state, f"❌ Error saving feedback: {str(e)}"

import anthropic
import openai
try:
    secrets = yaml.safe_load(open(args.secrets_file, "r"))
except FileNotFoundError:
    raise FileNotFoundError("secrets.yaml not found. Please create a secrets.yaml file in the same directory as this file with API keys for Anthropic and OpenAI.")

cost_dict = {
    "claude-3-7-sonnet-latest": {"input": 3/10**6, "output": 15/10**6},
}

class WebShopAgent:
    def __init__(self, args):
        self.args = args
        self.session_start_time = time.time()
        self.session_id = None
        self.agent_config = yaml.safe_load(open(args.agent_config_file, "r"))

        # These will be handled as transient properties
        self._env = None
        self._anthropic_client = None
        self._openai_client = None
        
        # State that can be serialized
        self.available_scenarios = []
        self.trajectory_start_time = None
        self.current_scenario = None
        self.task_completed = False
        self.trajectory = []
        self.trajectory_reward = None
        self.trajectory_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        self.num_dialog_turns = 0
        self.num_environment_steps = 0
        self.llm_cost = 0
        self.purchased_item = None

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Remove the unpicklable entries
        state['_env'] = None
        state['_anthropic_client'] = None
        state['_openai_client'] = None
        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
        # These will be recreated when needed via properties

    @property
    def env(self):
        if self._env is None:
            self.load_env()
        return self._env

    def load_env(self):
        self._env = WebAgentTextEnv(
            observation_mode="text_rich",
            human_goals=True,
            server=global_server,
            session_id=self.session_id if self.session_id is not None else "session_{}".format(time.time())
        )
        self.load_scenarios()
        logger.info("Loaded environment for session {} by copying global server".format(self.session_id))

    @property
    def anthropic_client(self):
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
        return self._anthropic_client

    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = openai.OpenAI(api_key=secrets['OPENAI_API_KEY'])
        return self._openai_client

    def load_scenarios(self):
        """Load scenarios and initialize environment if needed"""
        scenarios_file = self.args.scenarios_file
        try:
            with open(scenarios_file, "r") as f:
                scenarios = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Scenarios file {scenarios_file} not found.")
        self.available_scenarios = scenarios
        self._env.server.goals = scenarios
        random.seed(233)
        random.shuffle(self._env.server.goals)
        self._env.server.weights = [goal['weight'] for goal in scenarios]
        self._env.server.cum_weights = [0] + np.cumsum(self._env.server.weights).tolist()

    def initialize_environment(
            self, 
            scenario_index=None
        ):
        """Initialize the webshop environment"""
        try:
            # Initialize environment if not already initialized
            if self._env is None:
                self.load_env()
            self.available_scenarios = self._env.server.goals

            # Select scenario
            if scenario_index is None:# or scenario_index >= len(self.available_scenarios):
                random.seed(time.time())
                scenario_index = random.randint(0, len(self.available_scenarios) - 1)
            
            # Reset environment with selected scenario
            self.reset_agent()
            self._env.reset(scenario_index)
            self.current_scenario = self.available_scenarios[scenario_index]
            self.trajectory_start_time = time.time()
            logger.info(f"Initialized environment for session {self.session_id} with scenario {self.current_scenario['scenario_id']}")
            return True, self.current_scenario['instruction_text']

        except Exception as e:
            return False, f"Error initializing environment: {str(e)}"
    
    def get_agent_action(self) -> str:
        """
        Creates input messages based on system prompts and session trajectory,
        then queries the agent for an action
        """
        input_messages = []
        # input_messages.append({"role": "system", "content": self.agent_config["initial_system_prompt"]})
        for msg in self.trajectory:
            if isinstance(msg, str):
                input_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
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

        return agent_response.content[0].text


    def execute_and_generate_response(self, user_message: str) -> str:
        """Process user message and generate ReAct response"""
        self.trajectory.append({"role": "user", "content": f"USER SAYS: {user_message}"})
        observation_instruction_text = self._env.instruction_text.replace("Instruction:", "Instruction:\n")
        env_observation = self._env.observation.replace(observation_instruction_text, "")
        self.trajectory.append({"role": "user", "content": f"ENVIRONMENT OBSERVATION: {env_observation}"})

        num_errors = 0
        while True:
            try:
                agent_output = self.get_agent_action()
                assert "ACTION:" in agent_output, "Agent response must contain an action."
                action = agent_output.split("ACTION:")[-1].replace("\n", "").strip()
                self.trajectory.append({"role": "assistant", "content": agent_output})
                logger.info(f"Agent {self.session_id}: Action #{self.num_environment_steps}: {action}")

                # action is either (search/click/respond)[argument], parse accordingly
                if action.startswith("respond["):
                    response_text = action.split("respond[")[-1].strip("]").strip()
                    self.num_dialog_turns += 1
                    return response_text

                else:
                    # Execute the action in the environment
                    observation, reward, done, info = self._env.step(action)
                    observation = observation.replace(observation_instruction_text, "")
                    # Add observation to session trajectory
                    self.trajectory.append({"role": "user", "content": f"ENVIRONMENT OBSERVATION: {observation}"})
                    self.num_environment_steps += 1

                    if done:
                        self.task_completed = True
                        self.trajectory_reward = reward
                        self.purchased_item = info['purchased_item']
                        message = f"🎉 **I have bought the item!**\n\n"
                        message += f"**Product Name:** {self.purchased_item['product_name']}\n"
                        message += f"**Product Category**: {self.purchased_item['product_category']}\n"
                        message += "**Attributes:** {}\n".format(', '.join(self.purchased_item['attributes']))
                        if self.purchased_item["options"] != {}:
                            message += "**Selected Options:** {}\n".format(', '.join([f"{k}: {v}" for k, v in self.purchased_item['options'].items()]))
                        message += f"**Price:** {self.purchased_item['price']}\n"
                        return message

            except Exception as e:
                num_errors += 1
                if num_errors >= 5:
                    return f"Error encountered in execution: {str(e)}.\n\nPlease try rephrasing your request or reinitialize the environment."
                continue

            num_errors = 0

    def reset_agent(self):
        self.current_scenario = None
        self.task_completed = False
        self.trajectory = []
        self.trajectory_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        self.trajectory_reward = None
        self.num_dialog_turns = 0
        self.num_environment_steps = 0
        self.llm_cost = 0
        self.trajectory_start_time = None
        self.purchased_item = None
        logger.info(f"Reset agent for session {self.session_id}")
    
    def cleanup(self):
        """Clean up resources when session ends"""
        if self._env is not None:
            # Clean up environment
            del self._env
            self._env = None

        # Clear API clients
        self._anthropic_client = None
        self._openai_client = None
        self.reset_agent()
        # Clear session data
        self.available_scenarios = []
        logger.info(f"Cleaned up agent for session {self.session_id}")

def auto_initialize(session_agent, session_id):
    """Auto-initialize the environment on app load"""
    session_agent.session_start_time = time.time()
    session_agent.session_id = session_id
    status, chat_history, scenario_info, session_agent = initialize_session(-1, session_agent)
    logger.info(f"Initialized agent for session {session_id}")
    return gr.update(value=status), gr.update(value=chat_history), gr.update(value=scenario_info), session_agent

def initialize_session(scenario_index, session_agent):
    """Initialize a new webshop session"""
    scenario_idx = int(scenario_index) if scenario_index >= 0 else None
    if scenario_idx is not None and scenario_idx >= len(session_agent.available_scenarios):
        return f"❌ Invalid scenario index: {scenario_index}. Please choose a valid index between 0 and {len(session_agent.available_scenarios) - 1}, or -1 for random.", [], "Scenario not loaded :(", session_agent
    success, message = session_agent.initialize_environment(scenario_idx)
    
    if success:
        # Create welcome message
        welcome_msg = f"🛒 **WebShop Assistant Ready!**\n\n"
        welcome_msg += "I'm ready to help you shop! You can ask me to:\n"
        welcome_msg += "• Search for products\n"
        welcome_msg += "• Find specific items\n"
        welcome_msg += "• Compare options\n"
        welcome_msg += "• Add items to cart\n"
        welcome_msg += "• Complete purchases\n\n"
        welcome_msg += "What would you like to find today?"
        
        initial_chat = [gr.ChatMessage(role="assistant", content=welcome_msg)]
        return f"✅ Environment initialized successfully", initial_chat, message, session_agent
    else:
        return f"❌ Failed to initialize: {message}", [], "", session_agent


def chat_with_agent(user_message, chat_history, session_agent):
    """Process user message and generate agent response"""
    if not user_message.strip():
        return chat_history, "", session_agent

    try:
        # Add user message to chat history
        chat_history = chat_history + [
            gr.ChatMessage(role="user", content=user_message), 
            gr.ChatMessage(role="assistant", content="I am working on your request...")
        ]
        yield gr.update(interactive=False, value="", placeholder=""), chat_history, gr.update(interactive=False), session_agent, gr.update(is_visible=False)

        # Get agent response using the session's agent instance
        agent_response = session_agent.execute_and_generate_response(user_message)
        
        # Update chat history with agent's response
        chat_history[-1].content = agent_response.strip("\"")
        
    except Exception as e:
        agent_response = f"❌ **Error:** {str(e)}\n\nPlease try rephrasing your request or reinitialize the environment."
        chat_history[-1].content = agent_response
    
    # Return final state
    if "I have bought the item!" in agent_response:
        yield (
            gr.update(interactive=False, value="", placeholder="Please submit the feedback form, then start a new task. Do not repeat a task you have already completed."), 
            chat_history, 
            gr.update(interactive=False), 
            session_agent, 
            {"is_visible": True, "trajectory_id": session_agent.trajectory_id}
        )
    else:
        yield (
            gr.update(interactive=True, value="", placeholder="Ask me to search for products, find specific items, or help with shopping..."), 
            chat_history, 
            gr.update(interactive=True), 
            session_agent, 
            {"is_visible": False, "trajectory_id": None}
        )

# def get_session_info(session_agent):
#     """Get current session information"""
#     if session_agent._env is None:
#         return "No active session. Please initialize the environment first."
    
#     info = f"📊 **Session Status:**\n"
#     info += f"• Total number of agent actions: {session_agent.num_environment_steps + session_agent.num_dialog_turns} ({session_agent.num_environment_steps} environment steps + {session_agent.num_dialog_turns} dialog turns)\n"
#     info += f"• Task completed: {'✅' if session_agent.task_completed else '❌'}\n"
#     info += f"• Current scenario: {session_agent.current_scenario}\n\n"
    
#     return info


def save_session_data(annotator_name: str, session_agent, session_id):
    """Save current session data to a file"""
    if session_agent._env is None:
        return "No active session to save."

    if annotator_name.strip() == "":
        return "Please enter a valid annotator name to save the session data."

    filename = f"{args.save_dir}/annotator_{annotator_name.lower().replace(' ', '_')}-session_{session_id}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            session_data = json.load(f)
        assert ("annotator_name" in session_data \
                and "session_start_time" in session_data \
                and "trajectories" in session_data and isinstance(session_data["trajectories"], dict)
        ), f"Session data file {filename} is corrupted. Please delete the file and try again."
    else:
        session_data = {
            "annotator_name": annotator_name,
            "session_start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_agent.session_start_time)),
            "trajectories": {}
        }

    session_data["trajectories"][session_agent.trajectory_id] = {
        "scenario_id": session_agent.current_scenario['scenario_id'],
        "scenario": session_agent.current_scenario,
        "task_completed": session_agent.task_completed,
        "trajectory": session_agent.trajectory,
        "trajectory_reward": session_agent.trajectory_reward,
        "num_dialog_turns": session_agent.num_dialog_turns,
        "num_environment_steps": session_agent.num_environment_steps,
        "llm_model": session_agent.agent_config["model_name"],
        "llm_cost": round(session_agent.llm_cost, 4),
        "trajectory_time": (time.time() - session_agent.trajectory_start_time) / 60,
    }
    if session_agent.purchased_item is not None:
        session_data["trajectories"][session_agent.trajectory_id]["purchased_item"] = session_agent.purchased_item

    with open(filename, "w") as f:
        json.dump(session_data, f, indent=4)
    
    return "✅ Session data saved successfully."

def get_name(name: str):
    """Validate and process the annotator name input"""
    if not name or name.strip() == "":
        return None, "Please enter your name to continue"
    return name.strip(), None

def reset_session(session_agent):
    """Reset the current session"""
    session_agent.reset_agent()
    return "🔄 Session reset. Please initialize a new environment.", [], session_agent

def set_seed():
    """Set the seed to time.time() and then session_id to a random string"""
    random.seed(time.time())
    return ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=5))


# Create Gradio interface
with gr.Blocks(title="WebShop Agent", theme=gr.themes.Soft(), css="""
    * {
        font-family: 'Arial', 'Helvetica', sans-serif !important;
    }
    .gradio-container {
        font-family: 'Arial', 'Helvetica', sans-serif !important;
    }
    .large-font textarea {
        font-size: 1.2em !important;
    }
    /* Modal-like styling for instructions box */
    .instructions-box {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 1000 !important;
        opacity: 1 !important;
        background-color: var(--background-fill-primary) !important;
        border: 2px solid var(--border-color-primary) !important;
        padding: 2rem !important;
        border-radius: 10px !important;
        box-shadow: 0 0 10px rgba(0,0,0,0.25) !important;
        max-width: 800px !important;
        width: 90% !important;
        max-height: 90vh !important;
        overflow-y: auto !important;
    }
    .modal-overlay {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        background: rgba(0,0,0,0.5) !important;
        z-index: 999 !important;
    }
    /* Name input modal styling */
    .name-input-box {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 1001 !important;
        background-color: var(--background-fill-primary) !important;
        border: 2px solid var(--border-color-primary) !important;
        padding: 2rem !important;
        border-radius: 10px !important;
        box-shadow: 0 0 10px rgba(0,0,0,0.25) !important;
        max-width: 500px !important;
        width: 90% !important;
    }
    /* Satisfaction form styling */
    .satisfaction-form {
        background-color: var(--background-fill-primary) !important;
        border-left: 2px solid var(--border-color-primary) !important;
        padding: 1rem !important;
        height: 100% !important;
    }
""") as demo:
    
    session_agent = gr.State(WebShopAgent(args))
    session_id = gr.State("placeholder")
    show_instructions = gr.State(True)  # Add state for instructions visibility
    annotator_name = gr.State("")  # Add state for annotator name
    # Add state variables for satisfaction form
    satisfaction_form_state = gr.State({
        "is_visible": False,
        "trajectory_id": None
    })

    gr.Markdown("# Online Shopping with an AI Assistant")
    gr.Markdown("Chat with an AI shopping assistant can help you find and purchase products in the WebShop shopping website.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Environment Setup")

            scenario_index = gr.Number(
                value=-1, 
                label="Scenario Index (-1 for random)", 
                precision=0,
                minimum=-1,
                info="Choose a specific shopping scenario or random"
            )

            init_btn = gr.Button("🚀 Start New Shopping Task", variant="primary", size="lg")
            save_btn = gr.Button("💾 Save Session Data", variant="secondary")
            
            status_display = gr.Textbox(
                label="Session Status",
                lines=3,
                interactive=False,
                placeholder="Not initialized",
                value="Loading the environment..."
            )
        
        with gr.Column(scale=3):
            gr.Markdown("### 🛒 Current Shopping Scenario")
            scenario_info = gr.Textbox(
                value="Loading the scenario...",
                interactive=False,
                show_label=False,
                elem_classes=["large-font"],
            )

            gr.Markdown("### 💬 Chat with Shopping Assistant")
            
            chatbot = gr.Chatbot(
                label="User-Agent Conversation",
                height=300,
                show_label=False,
                avatar_images=["assets/person.png", "assets/robot.png"],
                value=[],
                type="messages"
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask me to search for products, find specific items, or help with shopping...",
                    lines=2,
                    scale=4,
                    interactive=False,
                )
                
                send_btn = gr.Button("💬 Send", variant="primary", scale=1, interactive=False)

        with gr.Column(scale=1, visible=False) as satisfaction_form:
            gr.Markdown("# 📝 Feedback Form")
            gr.Markdown("Please provide feedback about your shopping experience.")
            
            satisfied = gr.Textbox(
                label="Are you satisfied with the product?",
                info="Consider factors like price, features, and how well it matches your needs. Please provide detailed feedback.",
                lines=3,
                placeholder="Share your thoughts about the product..."
            )
            
            conforms = gr.Textbox(
                label="Does the product align with the scenario?",
                info="Explain whether and how this product fulfills the requirements from the shopping scenario.",
                lines=3,
                placeholder="Explain how well the product matches the scenario requirements..."
            )
            
            assistant_helpful = gr.Textbox(
                label="Did you find the assistant helpful?",
                info="Describe how well the assistant helped you find the right product. Was it efficient? Did it understand your needs? What could be improved?",
                lines=3,
                placeholder="Share your experience with the shopping assistant..."
            )
            
            submit_feedback = gr.Button("Submit Feedback", variant="primary", size="lg")

    # Create name input modal
    with gr.Column(visible=False, elem_classes="name-input-box") as name_input_container:
        gr.Markdown("# 👋 Welcome to WebShop!")
        gr.Markdown("Please enter your name or alias to begin.")
        name_input = gr.Textbox(
            label="Your Name",
            placeholder="Enter your name or alias",
        )
        name_error = gr.Markdown(visible=False, value="", elem_classes="error-text")
        name_submit = gr.Button("Continue", variant="primary")

    # Create instructions container
    with gr.Column(visible=False, elem_classes="instructions-box") as instructions_container:
        gr.Markdown("""
        # 📝 Instructions for WebShop Data Collection

        Welcome to the WebShop data collection interface! Please read these instructions carefully.

        In this game, you will use an AI shopping assistant to buy products for specific real-life scenarios.
        ### How It Works

        1. **You will receive a shopping scenario** (e.g. "You need a comfortable office chair for working from home").
        2. **Tell the AI assistant what you are looking for**: Give it a basic instruction like "I need an office chair"
        3. **The assistant searches and presents options**: The AI will explore the online store and describe products it finds, including details like price, features, and customer reviews.
        4. **Chat back and forth to find the right product**: The assistant may ask questions to help narrow down the best options for you. You can respond freely based on your preferences (e.g. stating your budget, preferred color, etc.), but make sure your responses are consistent with the scenario. You can be specific ("I need something under $200") or flexible ("I don't have a strong preference").
        5. **Buy the product when you are satisfied**: Once you find something that fits your scenario and preferences, tell the assistant you want to purchase it.

        ### Important Notes

        - **Read the scenario carefully, and be consistent with it**: Make sure that your instructions and messages to the assistant are consistent with all the details mentioned in the scenario.
        - **The assistant doesn't know your specific scenario**: It can only see what you tell it, not your full situation. So it might ask about your budget, preferences, or needs.
        - **You can only see what the assistant tells you**: You won't browse the store directly; the assistant acts as your eyes and ears.
        - **Think like you are really shopping**: Consider factors like price, color, size, and features when deciding whether a suggested product fits your needs.
        - **Please complete each shopping task fully**
        - If you want to start a new shopping task, click the reset button.


        ### Example Interaction
        - You: "I'm looking for a coffee maker"
        - Assistant: "I found several options. Are you looking for a single-serve machine or one that makes full pots? What's your budget range?"
        - You: "I'd like one that makes full pots, and I want to keep it under $100"
        - Assistant: "Great! I found a 12-cup programmable coffee maker for $79 with good reviews. It has an auto-shutoff feature and comes in black or white. Would you like to hear more about it?"
                    
        Continue chatting until you find the perfect product for your scenario!
        """)
        close_btn = gr.Button("Close Instructions", variant="primary")

    
    # Add help button at the top
    with gr.Row():
        help_btn = gr.Button("❓ Show Instructions")
        help_btn.click(
            fn=lambda x: not x,
            inputs=[show_instructions],
            outputs=[show_instructions],
            show_progress=False
        ).then(
            fn=lambda x: gr.update(visible=x),
            inputs=[show_instructions],
            outputs=[instructions_container]
        )

    # Add close button handler
    close_btn.click(
        fn=lambda: (False, gr.update(visible=False)),
        inputs=None,
        outputs=[show_instructions, instructions_container],
        show_progress=False
    )

    # Event handlers
    name_submit.click(
        fn=get_name,
        inputs=[name_input],
        outputs=[annotator_name, name_error]
    ).then(
        fn=lambda x: gr.update(visible=False) if x else gr.update(visible=True),
        inputs=[annotator_name],
        outputs=[name_input_container]
    ).then(
        fn=lambda: True,
        inputs=None,
        outputs=[show_instructions]
    ).then(
        fn=lambda x: gr.update(visible=x),
        inputs=[show_instructions],
        outputs=[instructions_container]
    )

    init_btn.click(
        fn=initialize_session,
        inputs=[scenario_index, session_agent],
        outputs=[status_display, chatbot, scenario_info, session_agent]
    ).then(
        fn=lambda: (
            gr.update(interactive=True, placeholder="Ask me to search for products, find specific items, or help with shopping..."), 
            gr.update(interactive=True)
        ),
        inputs=None,
        outputs=[user_input, send_btn]
    )

    save_btn.click(
        fn=save_session_data,
        inputs=[annotator_name, session_agent, session_id],
        outputs=status_display
    )
    
    msg_handlers = [
        user_input.submit,
        send_btn.click
    ]
    
    for handler in msg_handlers:
        handler(
            fn=chat_with_agent,
            inputs=[user_input, chatbot, session_agent],
            outputs=[user_input, chatbot, send_btn, session_agent, satisfaction_form_state],
            queue=True,
            show_progress=True,
        ).then(
            fn=save_session_data,
            inputs=[annotator_name, session_agent, session_id],
            outputs=status_display
        ).then(
            fn=lambda x: gr.update(visible=x["is_visible"]),
            inputs=[satisfaction_form_state],
            outputs=[satisfaction_form]
        )

    # Add handlers for satisfaction form
    submit_feedback.click(
        fn=submit_satisfaction_form,
        inputs=[satisfied, conforms, assistant_helpful, annotator_name, session_id, satisfaction_form_state],
        outputs=[satisfaction_form_state, status_display]
    ).then(
        fn=lambda x: gr.update(visible=x["is_visible"]),
        inputs=[satisfaction_form_state],
        outputs=[satisfaction_form]
    )
    
    # Auto-initialize session info on load
    demo.load(         # set seed to time.time() and then session_id to a random string
        fn=set_seed,
        inputs=None, 
        outputs=[session_id]
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=[name_input_container]
    ).then(             # initialize the session
        fn=auto_initialize,
        inputs=[session_agent, session_id],
        outputs=[status_display, chatbot, scenario_info, session_agent]
    ).then(             # make the user input and send button interactive
        fn=lambda x, y: (gr.update(interactive=True), gr.update(interactive=True)),
        inputs=[user_input, send_btn],
        outputs=[user_input, send_btn]
    )

    # Add cleanup to demo.close event
    demo.close = lambda: [agent.cleanup() for agent in [session_agent.value] if agent is not None]


if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=4000,
        show_error=True,
        debug=True
    )