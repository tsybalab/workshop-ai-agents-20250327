"""Assignment 2: Multi-Agent Conversation
Implement a system with two AI agents using AutoGen that can have a conversation to solve a problem.

Features:
1. Automatic conversation termination when a solution is found
2. Solution evaluation based on specific criteria
3. Detailed feedback on proposed solutions
"""

import os
from typing import Dict, List, Optional

import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EvaluatingUser(UserProxyAgent):
    """A user agent that evaluates solutions and automatically terminates conversations.

    This agent extends UserProxyAgent to add solution evaluation capabilities:
    - Tracks and evaluates 5 key criteria in proposed solutions:
        1. Library suggestions (specific libraries mentioned)
        2. Interactive features (zoom, hover, click functionality)
        3. Ease of use (user-friendliness, simplicity)
        4. Code examples (practical implementation)
        5. Pros and cons analysis

    Auto-termination occurs in two cases:
    1. Solution found: When at least 3 out of 5 criteria are met
    2. Stale conversation: When max_stale_turns is reached without new criteria being met

    The agent provides detailed feedback on missing criteria when terminating due to
    stale turns, helping guide the conversation towards a complete solution.

    Args:
        max_stale_turns (int): Maximum allowed turns without progress (default: 3)
        *args: Additional positional arguments passed to UserProxyAgent
        **kwargs: Additional keyword arguments passed to UserProxyAgent
    """

    def __init__(self, *args, max_stale_turns: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.solution_found = False
        self.evaluation_criteria = {
            "library_suggestion": False,  # Specific library mentioned
            "interactivity": False,  # Interactive features discussed
            "ease_of_use": False,  # Ease of use addressed
            "examples": False,  # Code examples provided
            "pros_cons": False,  # Pros and cons discussed
        }
        self.evaluation_history = []
        self.max_stale_turns = max_stale_turns
        self.stale_turn_count = 0
        self.last_criteria_met = set()

    def evaluate_solution(self, message: str) -> Dict:
        """Evaluate a proposed solution based on multiple criteria."""
        criteria = self.evaluation_criteria.copy()

        # Check for library suggestions
        libraries = ["plotly", "altair", "bokeh", "seaborn", "matplotlib"]
        criteria["library_suggestion"] = any(
            lib in message.lower() for lib in libraries
        )

        # Check for interactivity discussion
        criteria["interactivity"] = any(
            term in message.lower()
            for term in ["interactive", "dynamic", "zoom", "hover", "click"]
        )

        # Check for ease of use discussion
        criteria["ease_of_use"] = any(
            term in message.lower()
            for term in [
                "easy",
                "simple",
                "beginner",
                "straightforward",
                "user-friendly",
            ]
        )

        # Check for code examples
        criteria["examples"] = "```" in message or "example:" in message.lower()

        # Check for pros and cons
        criteria["pros_cons"] = (
            "advantage" in message.lower()
            or "disadvantage" in message.lower()
            or "pro" in message.lower()
            or "con" in message.lower()
        )

        # Calculate score
        score = sum(criteria.values())
        evaluation = {
            "score": score,
            "max_score": len(criteria),
            "criteria_met": criteria,
            "is_solution": score >= 3,  # Need at least 3 criteria met
        }

        # Check for progress (new criteria being met)
        current_criteria_met = {k for k, v in criteria.items() if v}
        new_criteria = current_criteria_met - self.last_criteria_met

        if new_criteria:
            # Reset stale counter if we met new criteria
            self.stale_turn_count = 0
        else:
            # Increment stale counter if no new criteria were met
            self.stale_turn_count += 1

        # Update last criteria met
        self.last_criteria_met = current_criteria_met

        self.evaluation_history.append(evaluation)
        return evaluation

    def format_evaluation(self, eval_result: Dict) -> str:
        """Format the evaluation results into a readable message."""
        criteria_symbols = {True: "✅", False: "❌"}

        message = f"\nSolution Evaluation (Score: {eval_result['score']}/{eval_result['max_score']}):\n"
        for criterion, met in eval_result["criteria_met"].items():
            message += (
                f"{criteria_symbols[met]} {criterion.replace('_', ' ').title()}\n"
            )

        if eval_result["is_solution"]:
            message += "\n🎯 This is a satisfactory solution!\n"
        else:
            message += (
                "\n🔄 More details needed. Please provide additional information.\n"
            )

        return message

    def on_receive(
        self, messages: List[Dict], sender: Agent, config: Optional[Dict] = None
    ) -> bool:
        """Process received messages and evaluate solutions.
        Returns True if the conversation should continue, False to terminate."""
        if messages:
            last_message = messages[-1]["content"]
            evaluation = self.evaluate_solution(last_message)

            # Print evaluation
            print(self.format_evaluation(evaluation))

            # Update solution status
            self.solution_found = evaluation["is_solution"]

            # Check termination conditions
            if self.solution_found or self.stale_turn_count >= self.max_stale_turns:
                return False

        return True

    def generate_reply(
        self,
        messages: List[Dict],
        sender: Optional[Agent] = None,
        config: Optional[Dict] = None,
    ) -> Dict:
        """Handle replies and terminate if a solution is found or timeout reached."""
        # First check if we should terminate
        should_continue = self.on_receive(messages, sender, config)
        if not should_continue:
            if self.solution_found:
                return {
                    "content": "👍 Thank you! This solution meets all my requirements. Let's end our discussion here.",
                    "terminate": True,
                }
            else:  # Must be stale turns timeout
                # Calculate missing criteria
                missing_criteria = [
                    k.replace("_", " ").title()
                    for k, v in self.evaluation_criteria.items()
                    if not v
                ]
                missing_list = "\n- ".join(missing_criteria)

                return {
                    "content": f"""🕛 Time to wrap up. We've had {self.stale_turn_count} turns without progress.

Still missing information about:
- {missing_list}

Let's end here and try a different approach.""",
                    "terminate": True,
                }

        # Continue the conversation
        return super().generate_reply(messages=messages, sender=sender, config=config)


def get_config_list():
    """
    Get the LLM configuration based on the provider.

    Returns a list of configuration dictionaries for the specified LLM provider.
    The provider is determined by the LLM_PROVIDER environment variable.

    Supported providers:
    - openai: Uses GPT-4o-mini model

    Returns:
        List[Dict]: List of configuration dictionaries for the LLM

    Raises:
        ValueError: If LLM_PROVIDER is not set or unsupported,
                  or if required API key is missing
    """
    llm_provider = os.getenv("LLM_PROVIDER")
    if not llm_provider:
        raise ValueError("LLM_PROVIDER environment variable not set")

    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        config = {
            "model": "gpt-4o-mini",
            "api_key": api_key,
        }
        return [config]
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def setup_agents():
    """
    Set up the user and assistant agents.
    """
    # Configure the LLM
    config_list = get_config_list()

    # Create the assistant agent with detailed instructions
    assistant = AssistantAgent(
        name="Assistant",
        llm_config={"config_list": config_list},
        system_message="""You are a data visualization expert. Follow these guidelines:
1. Ask clarifying questions about requirements if needed
2. When suggesting solutions:
   - Recommend specific libraries with clear reasoning
   - Discuss interactive features and capabilities
   - Explain ease of use and learning curve
   - Provide brief code examples when relevant
   - Compare pros and cons of different options
3. Be thorough but concise""",
    )

    # Create the evaluating user agent
    user = EvaluatingUser(
        name="User",
        llm_config={"config_list": config_list},
        human_input_mode="NEVER",  # This ensures the agent runs autonomously
        system_message="You are a user seeking data visualization solutions. Evaluate proposed solutions based on:"
        "\n- Library suggestions"
        "\n- Interactive features"
        "\n- Ease of use"
        "\n- Code examples"
        "\n- Pros and cons analysis",
        code_execution_config=False,  # Disable code execution for this agent
    )

    return user, assistant


def define_user_problem():
    """
    Define the initial problem that the user agent will present.
    """
    # Define a specific problem for the user agent
    problem = """
    We have approximately 100,000 sales records and static data is sufficient.
    We want line charts for monthly trends, bar charts for regional sales, and pie charts for product categories.
    The dashboard will be a web application deployed with Flask.
    The users are sales and marketing staff with limited technical skills, so it must be user-friendly.
    Can you recommend the most suitable Python visualization library and explain why?
    """

    return problem


def run_conversation():
    """
    Run the conversation between the user and assistant agents.
    """
    try:
        # Set up the agents
        user, assistant = setup_agents()

        # Define the initial problem
        problem = define_user_problem()

        # Start the conversation with the user's problem
        chat_result = user.initiate_chat(
            assistant,
            message=problem,
            max_turns=10,  # Limit the conversation to 10 turns
        )

        # Print final evaluation summary
        if isinstance(user, EvaluatingUser) and user.evaluation_history:
            print("\n=== Final Evaluation Summary ===")
            best_evaluation = max(user.evaluation_history, key=lambda x: x["score"])
            print(user.format_evaluation(best_evaluation))
            print(f"Total turns: {len(user.evaluation_history)}")

        return chat_result
    except KeyboardInterrupt:
        print("\n\n👋 Exiting gracefully. Bye!")
        return {"status": "interrupted"}


if __name__ == "__main__":
    run_conversation()
