"""
Assignment 1: Basic Agent Setup
A simple AI agent using LangGraph that can answer questions about a specific topic.

TODO:
1. Improve prompt: add guardrails to focus the conversation on the topic
2. Add cycle to the graph, instead of using "while True" outside of the graph
3. Add DuckDuckGo search tool to improve the agent's responses, see https://python.langchain.com/docs/integrations/tools/ddg/
4. (optional) Use Google's gemini-2.0-flash-lite instead of OpenAI, see https://aistudio.google.com/apikey and https://python.langchain.com/docs/integrations/chat/google_generative_ai/
"""

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """Defines the structure of the agent's state."""

    messages: list
    topic: str
    user_input: str


def get_llm():
    """Initialize and return the appropriate LLM based on the provider."""
    llm_provider = os.getenv("LLM_PROVIDER")
    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.7,
        )
    elif llm_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            api_key=api_key,
            temperature=0.7,
        )
    else:
        raise ValueError("Unsupported LLM provider")


def get_user_input(state: AgentState) -> AgentState:
    """Prompt user for input and update the state."""
    user_input = input("\nYou: ").strip()
    state["user_input"] = user_input
    if user_input and user_input.lower() != "exit":
        state["messages"].append({"role": "user", "content": user_input})
    return state


def process_input(state: AgentState) -> AgentState:
    """Process the input using the LLM and update the state with the response."""
    topic = state["topic"]
    llm = get_llm()
    system_message = (
        f"You are an expert on '{topic}'. "
        "Provide accurate and helpful information strictly related to this topic. "
        "If you don't know the answer, admit it clearly and never make up information. "
        "Avoid speculation, imaginary answers, or going off-topic. "
        "If the user strays, politely guide them back to the main subject. "
    )
    response = llm.invoke(
        [{"role": "system", "content": system_message}, *state["messages"]]
    )
    state["messages"].append({"role": "assistant", "content": response.content})
    print(f"\nAgent: {response.content}")
    return state


def check_continue(state: AgentState) -> Literal["continue", "__end__"]:
    """Check if the conversation should continue or end."""
    return "continue" if state["user_input"].lower() != "exit" else END


def create_agent_graph(topic: str = "Python programming"):
    """Create and compile a LangGraph state machine for the agent."""
    workflow = StateGraph(AgentState)
    workflow.add_node("get_user_input", get_user_input)
    workflow.add_node("process_input", process_input)
    workflow.add_conditional_edges(
        "get_user_input", check_continue, {"continue": "process_input", "__end__": END}
    )
    workflow.add_edge("process_input", "get_user_input")
    workflow.set_entry_point("get_user_input")
    return workflow.compile()


def main():
    """Run the agent in a loop, interacting with the user until 'exit' is typed."""
    topic = "Python programming"
    agent = create_agent_graph(topic)
    print(f"Agent specialized in {topic} is ready. Type 'exit' to quit.")
    state = AgentState(messages=[], topic=topic, user_input="")
    try:
        agent.invoke(state)
    except KeyboardInterrupt:
        print("\n\n👋 Exiting gracefully. Bye!")


if __name__ == "__main__":
    main()
