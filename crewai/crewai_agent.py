"""
Assignment 3: Specialized Crew Creation
Use CrewAI to create a specialized team of agents with different roles that collaborate to produce content.

TODO:
1. Configure the researcher agent with appropriate tools
2. Make sure that final article is an article, without any additional text and without critique
"""

import os

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


# Define agent roles


def create_researcher():
    return Agent(
        role="Researcher",
        goal="Gather information",
        backstory="An expert in data collection",
        llm=ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            max_tokens=1000,
            request_timeout=60,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )


def create_writer():
    return Agent(
        role="Writer",
        goal="Transform research into an article",
        backstory="A skilled writer",
        llm=ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )


# Define tasks


def define_tasks(researcher, writer, topic):
    research_task = Task(
        description=f"Research topic: {topic}. Include definitions, use cases, and trends.",
        expected_output="Structured research notes.",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a final article based on the research. Only output the article — no notes or feedback.",
        expected_output="An article in markdown format, approx. 500 words, without meta-comments or explanations.",
        agent=writer,
        context=[research_task],  # ✅ keep it as object, not dict
    )

    return [research_task, writing_task]


# Execute the crew


def execute_crew(topic):
    researcher = create_researcher()
    writer = create_writer()
    tasks = define_tasks(researcher, writer, topic)

    crew = Crew(agents=[researcher, writer], tasks=tasks, verbose=True)
    result = crew.kickoff()
    print("\n\n=== FINAL ARTICLE ===\n\n")
    print(result)


def main():
    topic = "The Role of AI in Modern Healthcare"
    execute_crew(topic)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting gracefully. Bye!")
