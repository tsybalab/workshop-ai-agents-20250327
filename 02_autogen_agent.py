"""
Assignment 2: Multi-Agent Conversation
Implement a system with two AI agents using AutoGen that can have a conversation to solve a problem.

TODO:
1. Make agents stop after solving the problem
"""

import os
from dotenv import load_dotenv
import autogen
from autogen import Agent, ConversableAgent, UserProxyAgent, AssistantAgent

# Load environment variables
load_dotenv()

def setup_agents():
    """
    Set up the user and assistant agents.
    """
    # Configure the LLM
    config_list = [
        {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ]
    
    # Create the assistant agent
    assistant = AssistantAgent(
        name="Assistant",
        llm_config={"config_list": config_list},
        system_message="You are a helpful AI assistant. Your goal is to solve the user's problem by asking clarifying questions and providing solutions.",
    )
    
    # Create the user agent with a specific problem
    user = UserProxyAgent(
        name="User",
        llm_config={"config_list": config_list},
        human_input_mode="NEVER",  # This ensures the agent runs autonomously
        system_message="You are a user with a specific problem. Provide details about your problem when asked and evaluate the assistant's solutions.",
        code_execution_config=False,  # Disable code execution for this agent
    )
    
    return user, assistant

def define_user_problem():
    """
    Define the initial problem that the user agent will present.
    """
    # Define a specific problem for the user agent
    problem = """
    I'm trying to build a data visualization dashboard for my company's sales data, 
    but I'm having trouble deciding which visualization library to use in Python. 
    I need something that's easy to use but also powerful enough for interactive visualizations.
    """
    
    return problem

def run_conversation():
    """
    Run the conversation between the user and assistant agents.
    """
    # Set up the agents
    user, assistant = setup_agents()
    
    # Define the initial problem
    problem = define_user_problem()
    
    # Start the conversation with the user's problem
    user.initiate_chat(
        assistant,
        message=problem,
        max_turns=10  # Limit the conversation to 10 turns
    )

if __name__ == "__main__":
    run_conversation()
