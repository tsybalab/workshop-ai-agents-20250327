"""
Assignment 5: Browser Use

TODO:
1. Implement your example
"""

import asyncio
import os

from browser_use import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize our LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)


# Example 1: Get information from a website
async def example_1():
    print("\nExample 1: Get information from Python.org")
    agent = Agent(
        task="Visit https://www.python.org and tell me what's on the homepage",
        llm=llm,
    )
    await agent.run()
    print("Task completed!")


# Example 2: Compare information from two websites
async def example_2():
    print("\nExample 2: Compare Python versions")
    agent = Agent(
        task="Compare the latest Python version mentioned on python.org with the latest version mentioned on Wikipedia's Python page",
        llm=llm,
    )
    await agent.run()
    print("Task completed!")


# Example 3: Extract links and navigate
async def example_3():
    print("\nExample 3: Find documentation links")
    agent = Agent(
        task="Go to https://www.python.org, find a link to the Python documentation, and tell me what topics are covered in the documentation",
        llm=llm,
    )
    await agent.run()
    print("Task completed!")


async def main():
    print("Browser-Use Agent Demo")
    print("======================")

    # await example_1()
    # await example_2()
    await example_3()

    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())
