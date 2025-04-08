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
from langchain_community.tools import DuckDuckGoSearchRun
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
    search_results: str


def get_llm():
    """Initialize and return the appropriate LLM based on the provider."""
    llm_provider = os.getenv("LLM_PROVIDER")
    if not llm_provider:
        raise ValueError("LLM_PROVIDER environment variable not set")

    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.7,
        )
    elif llm_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            api_key=api_key,
            temperature=0.7,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def get_user_input(state: AgentState) -> AgentState:
    """Prompt user for input and update the state."""
    user_input = input("\nYou: ").strip()
    state["user_input"] = user_input
    if user_input and user_input.lower() != "exit":
        state["messages"].append({"role": "user", "content": user_input})
    return state


def search_ddg(state: AgentState) -> AgentState:
    """Search DuckDuckGo for relevant information."""
    search = DuckDuckGoSearchRun()
    query = f"{state['topic']} {state['user_input']}"
    try:
        print("\n🔍 Searching for:", query)
        results = search.run(query)
        if results.strip():
            print("✨ Found relevant information")
            state["search_results"] = results
            # Format search results for better readability and citation
            formatted_results = (
                f"[SEARCH_RESULTS]\n"
                f"Query: '{query}'\n\n"
                f"Results:\n{results}\n\n"
                f"Instructions: When using this information, quote relevant parts and cite as (from search results)"
            )
            state["messages"].append({"role": "system", "content": formatted_results})
        else:
            print("❌ No relevant information found")
            state["search_results"] = ""
    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        state["search_results"] = ""
    return state


def process_input(state: AgentState) -> AgentState:
    """Process the input using the LLM and update the state with the response."""
    topic = state["topic"]
    llm = get_llm()
    system_message = (
        f"You are an expert on '{topic}'. Follow these rules strictly:\n"
        "1. ALWAYS analyze any search results provided (marked with [SEARCH_RESULTS])\n"
        "2. When using search results:\n"
        "   - Search results are trusted and current\n"
        "   - Start with '🔍 Based on my search...'\n"
        "   - Quote specific information using quotation marks\n"
        "   - After each quote, cite it as (from search results)\n"
        "   - Add your expert analysis and additional context\n"
        "3. When NOT using search results:\n"
        "   - Start with '💡 From my knowledge...'\n"
        "   - Explain concepts clearly using your expertise\n"
        "4. If search results are irrelevant or low quality:\n"
        "   - Say 'The search results weren't relevant here...'\n"
        "   - Then provide your expert knowledge instead\n"
        "5. Always maintain a natural, conversational tone\n"
        "6. If you're unsure, admit it clearly"
    )

    # Start with system message and all conversation history (includes search results)
    messages = [{"role": "system", "content": system_message}]
    messages.extend(state["messages"])

    response = llm.invoke(messages)
    state["messages"].append({"role": "assistant", "content": response.content})
    print(f"\nAgent: {response.content}")
    return state


def check_continue(state: AgentState) -> Literal["continue", "__end__"]:
    """Check if the conversation should continue or end."""
    return "continue" if state["user_input"].lower() != "exit" else END


def create_agent_graph(topic: str = "Python programming"):
    """Create and compile a LangGraph state machine for the agent."""
    workflow = StateGraph(AgentState)

    # Add nodes to the workflow
    workflow.add_node("get_user_input", get_user_input)
    workflow.add_node("search_ddg", search_ddg)  # Search functionality
    workflow.add_node("process_input", process_input)

    # Define the flow
    workflow.add_conditional_edges(
        "get_user_input", check_continue, {"continue": "search_ddg", "__end__": END}
    )
    workflow.add_edge("search_ddg", "process_input")
    workflow.add_edge("process_input", "get_user_input")

    workflow.set_entry_point("get_user_input")
    return workflow.compile()


def main() -> AgentState:
    """Run the agent in a loop, interacting with the user until 'exit' is typed.

    Returns:
        AgentState: The final state of the agent after completion or interruption.
    """
    topic = "Python programming"
    agent = create_agent_graph(topic)
    print(f"Agent specialized in {topic} is ready. Type 'exit' to quit.")
    state = AgentState(messages=[], topic=topic, user_input="", search_results="")
    final_state = agent.invoke(state)
    return final_state


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting gracefully. Bye!")
