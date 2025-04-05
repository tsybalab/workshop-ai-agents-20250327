"""
Assignment 5: Browser Use

TODO:
1. Implement your example
"""

import asyncio
import os
import time

from browser_use import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


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


async def run_agent_with_retries(agent, retries=3, base_delay=3):
    for attempt in range(1, retries + 1):
        try:
            await agent.run()
            return
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt == retries:
                raise
            delay = base_delay * attempt
            print(f"⏳ Retrying in {delay} seconds...")
            time.sleep(delay)


# Example 1: Get information from a website
async def example_1():
    print("\nExample 1: Get information from Python.org")
    agent = Agent(
        task="Visit https://www.python.org and tell me what's on the homepage",
        llm=get_llm(),
    )
    await run_agent_with_retries(agent)
    print("Task completed!")


# Example 2: Compare information from two websites
async def example_2():
    print("\nExample 2: Compare Python versions")
    agent = Agent(
        task="Compare the latest Python version mentioned on python.org with the latest version mentioned on Wikipedia's Python page",
        llm=get_llm(),
    )
    await run_agent_with_retries(agent)
    print("Task completed!")


# Example 3: Extract links and navigate
async def example_3():
    print("\nExample 3: Find documentation links")
    agent = Agent(
        task="Go to https://www.python.org, find a link to the Python documentation, and tell me what topics are covered in the documentation",
        llm=get_llm(),
    )
    await run_agent_with_retries(agent)
    print("Task completed!")


# Example 4: Search specific medical AI topic on DuckDuckGo
async def example_4():
    print(
        "\nExample 4: Explore AI impact on medical diagnostic accuracy via DuckDuckGo"
    )
    agent = Agent(
        task=(
            "Go to https://www.nih.gov. In the search bar type 'AI in medical diagnostic accuracy' and press Enter. "
            "From the first 7 results or fewer if less are available, extract the title, link, and abstract (if available). "
            "Summarize the results. Output the final summary as markdown."
        ),
        llm=get_llm(),
    )
    await run_agent_with_retries(agent)
    print("Task completed!")


async def main():
    print("Browser-Use Agent Demo")
    print("======================")

    # await example_1()
    # await example_2()
    # await example_3()
    await example_4()

    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())
