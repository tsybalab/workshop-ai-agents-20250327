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

# Load environment variables
load_dotenv()


def create_researcher_agent():
    """
    Create a researcher agent that gathers information on a topic.
    """
    researcher = Agent(
        role="Research Specialist",
        goal="Conduct thorough research on technical topics and provide accurate information",
        backstory="You are an expert researcher with a background in computer science and technical writing. "
        "You excel at finding relevant and accurate information on complex technical topics.",
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini",
    )

    return researcher


def create_writer_agent():
    """
    Create a writer agent that produces content based on research.
    """
    writer = Agent(
        role="Technical Writer",
        goal="Transform research into clear, engaging, and informative content",
        backstory="You are a skilled technical writer who specializes in explaining complex concepts "
        "in accessible language. You have a talent for organizing information logically "
        "and creating engaging narratives around technical topics.",
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini",
    )

    return writer


def create_critic_agent():
    """
    Create a critic agent that reviews and improves content.
    """
    critic = Agent(
        role="Content Critic",
        goal="Evaluate content for accuracy, clarity, and engagement, providing constructive feedback",
        backstory="You are a meticulous editor with an eye for detail and a commitment to quality. "
        "You have extensive experience reviewing technical content and can identify areas "
        "for improvement in both substance and style.",
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini",
    )

    return critic


def create_research_task(researcher, topic):
    """
    Create a task for the researcher agent.
    """
    research_task = Task(
        description=f"Research the topic: {topic}. Gather key information, including:\n"
        f"1. Main concepts and definitions\n"
        f"2. Historical context and development\n"
        f"3. Current state and applications\n"
        f"4. Future trends and challenges\n\n"
        f"Compile your findings into a structured research report with clear sections.",
        agent=researcher,
        expected_output="A comprehensive research report on the topic with clear sections covering "
        "key concepts, history, current applications, and future trends.",
    )

    return research_task


def create_writing_task(writer, topic):
    """
    Create a task for the writer agent.
    """
    writing_task = Task(
        description=f"Using the research provided, write a 500-word article on {topic}. The article should:\n"
        f"1. Have an engaging introduction that hooks the reader\n"
        f"2. Explain key concepts clearly with examples\n"
        f"3. Be organized with logical flow and appropriate headings\n"
        f"4. Include a conclusion that summarizes key points and implications\n\n"
        f"Focus on making the content accessible to a technical audience with some background knowledge.",
        agent=writer,
        expected_output="A well-structured 500-word article on the topic that is engaging, "
        "informative, and accessible to a technical audience.",
        context=[],  # This will be filled with the research task output
    )

    return writing_task


def create_critique_task(critic, topic):
    """
    Create a task for the critic agent.
    """
    critique_task = Task(
        description=f"Review the article on {topic} and provide constructive feedback. Evaluate the article based on:\n"
        f"1. Technical accuracy and depth\n"
        f"2. Clarity and organization\n"
        f"3. Engagement and readability\n"
        f"4. Completeness of coverage\n\n"
        f"Provide specific suggestions for improvement and edit the article to address these issues.",
        agent=critic,
        expected_output="A detailed critique of the article with specific feedback and an "
        "improved version of the article that addresses the identified issues.",
        context=[],  # This will be filled with the writing task output
    )

    return critique_task


def create_specialized_crew(topic):
    """
    Create a specialized crew to produce content on a topic.
    """
    # Create the agents
    researcher = create_researcher_agent()
    writer = create_writer_agent()
    critic = create_critic_agent()

    # Create the tasks
    research_task = create_research_task(researcher, topic)
    writing_task = create_writing_task(writer, topic)
    critique_task = create_critique_task(critic, topic)

    # Set up task dependencies
    writing_task.context = [research_task]
    critique_task.context = [writing_task]

    # Create the crew
    crew = Crew(
        agents=[researcher, writer, critic],
        tasks=[research_task, writing_task, critique_task],
        verbose=True,
    )

    return crew


def main():
    # Set the topic for the article
    topic = "The Role of Large Language Models in Modern Software Development"

    # Create the specialized crew
    crew = create_specialized_crew(topic)

    # Run the crew to produce the article
    result = crew.kickoff()

    # Print the final result
    print("\n\n=== FINAL ARTICLE ===\n\n")
    print(result)


if __name__ == "__main__":
    main()
