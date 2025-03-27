"""
Assignment 1: Basic Agent Setup
A simple AI agent using LangGraph that can answer questions about a specific topic.
    
TODO: Complete this code to:
1. Use Google's gemini-2.0-flash-lite instead of OpenAI, see https://aistudio.google.com/apikey and https://python.langchain.com/docs/integrations/chat/google_generative_ai/ 
2. Add cycle to the graph, instead of using "while True" outside of the graph
3. Improve prompt: add guardrails to focus the conversation on the topic
4. Add DuckDuckGo search tool to improve the agent's responses, see https://python.langchain.com/docs/integrations/tools/ddg/
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# Load environment variables
load_dotenv()

# Define the state structure
class AgentState(TypedDict):
    messages: list
    topic: str

# Initialize the LLM
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )

# Process input and generate response
def process_input(state: AgentState) -> AgentState:
    """
    Process the user input and generate a response.
    """
    # Get the latest user message
    # latest_message = state["messages"][-1]
    
    # Get the topic specialization
    topic = state["topic"]
    
    # Initialize the LLM
    llm = get_llm()
    
    # Create a system message with the topic specialization
    system_message = f"You are an expert on {topic}. Provide accurate and helpful information about this topic."
    
    # Generate a response
    response = llm.invoke([
        {"role": "system", "content": system_message},
        *state["messages"]
    ])
    
    # Add the assistant's response to the messages
    state["messages"].append({"role": "assistant", "content": response.content})
    
    return state

# Define the graph
def create_agent_graph(topic: str = "Python programming"):
    """
    Create a LangGraph state machine for the agent.
    """
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add the process_input node
    workflow.add_node("process_input", process_input)
    
    # Add edges - in this simple case, there should be no edges
    # workflow.add_edge("process_input", "process_input")
    workflow.set_entry_point("process_input")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Create a simple interface to interact with the agent
def main():
    # Choose a topic for your agent
    topic = "Python programming"  # You can change this to any topic
    
    # Create the agent graph
    agent = create_agent_graph(topic)
    
    print(f"Agent specialized in {topic} is ready. Type 'exit' to quit.")
    
    # Initialize the state
    state = {
        "messages": [],
        "topic": topic
    }
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            break
        
        # Add the user message to the state
        state["messages"].append({"role": "user", "content": user_input})
        
        # Run the agent
        state = agent.invoke(state)
        
        # Print the agent's response
        print(f"\nAgent: {state['messages'][-1]['content']}")

if __name__ == "__main__":
    main()
