import os
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GKey")

# 2. Define the Groq LLM using CrewAI's native LLM wrapper

groq_llm = LLM(model="groq/openai/gpt-oss-120b", temperature=0.3)


# 3. Define Tools using CrewAI's @tool decorator
@tool("Get Weather")
def get_weather(city: str) -> str:
  """Get current weather conditions for a given city."""
  weather_dict = {
      "lahore": "Lahore: 32°C, sunny",
      "karachi": "Karachi: 31°C, humid",
      "islamabad": "Islamabad: 28°C, clear",
  }
  return weather_dict.get(city.lower(), f"{city}: 30°C, fair")


@tool("Convert Currency")
def convert_currency(amount: float, to: str) -> str:
  """Convert an amount from PKR to target currencies (USD, EUR, GBP)."""
  rates = {"usd": 278.0, "eur": 300.0, "gbp": 350.0}
  target = to.lower()
  if target in rates:
    converted = amount / rates[target]
    return f"PKR {amount:,.2f} = {target.upper()} {converted:,.2f}"
  return f"Unsupported currency: {to}"


# 4. Define Agents
budget_agent = Agent(
    role="Budget Analyst",
    goal=(
        "Analyze travel requirements, check weather or currency using tools"
        " when needed, and produce a detailed cost split in PKR."
    ),
    backstory=(
        "Expert financial planner and travel budget specialist who ensures"
        " trips stay strictly within financial limits using precise calculations"
        " and data tools."
    ),
    tools=[get_weather, convert_currency],
    llm=groq_llm,
    verbose=True,
)

writer_agent = Agent(
    role="Travel Itinerary Writer",
    goal=(
        "Turn budget breakdowns and trip details into a clear, structured"
        " day-by-day travel plan."
    ),
    backstory=(
        "Professional travel writer and guide who creates engaging, practical,"
        " and well-organized itineraries aligned with user budgets."
    ),
    llm=groq_llm,
    verbose=True,
)

reviewer_agent = Agent(
    role="Itinerary Reviewer",
    goal=(
        "Critically review the trip itinerary and budget for consistency,"
        " correctness, and viability."
    ),
    backstory=(
        "Meticulous quality control editor who evaluates travel plans, checks"
        " for budget overflows or errors, and ensures high quality before final"
        " approval."
    ),
    llm=groq_llm,
    verbose=True,
)


# 5. Define Tasks
budget_task = Task(
    description=(
        "Analyze the travel goal: '{goal}'. Use tools to check weather or"
        " convert currency if needed, and produce a detailed cost breakdown in"
        " PKR covering accommodation, food, transport, and entries."
    ),
    expected_output=(
        "A structured cost breakdown table in PKR detailing all expense"
        " categories."
    ),
    agent=budget_agent,
)

writing_task = Task(
    description=(
        "Using the budget breakdown provided by the Budget Analyst, write a"
        " comprehensive 3-day day-by-day itinerary for the goal: '{goal}'."
        " Specify activities, dining options, and transport suggestions."
    ),
    expected_output=(
        "A structured, engaging day-by-day travel itinerary matching the budget"
        " constraints."
    ),
    agent=writer_agent,
)

review_task = Task(
    description=(
        "Critically review the generated travel plan and budget breakdown for"
        " the goal: '{goal}'. Verify numbers add up and everything aligns."
        " Output 'APPROVED' if perfect, or list specific fixes if changes are"
        " needed."
    ),
    expected_output=(
        "An approval statement or a concise list of corrections required for"
        " the itinerary."
    ),
    agent=reviewer_agent,
)


# 6. Assemble the Crew and Define Process Flow
travel_crew = Crew(
    agents=[budget_agent, writer_agent, reviewer_agent],
    tasks=[budget_task, writing_task, review_task],
    process=Process.sequential,  
    verbose=True,
)


if __name__ == "__main__":
  inputs = {"goal": "3 days in Lahore for 2 people, budget PKR 60,000"}

  print(f"=== Starting CrewAI Trip Orchestration for: '{inputs['goal']}' ===\n")
  result = travel_crew.kickoff(inputs=inputs)

  print("\n=== FINAL APPROVED TRIP PLAN ===")
  print(result)