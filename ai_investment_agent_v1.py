from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.playground import Playground, serve_playground_app

# Initialize AI Model using OpenAI
llm = OpenAIChat(model="gpt-4")

# Set up storage to remember user interactions
storage = SqliteAgentStorage(db_path="agents.db")

# Define tools for stock data and financial news
stock_tool = YFinanceTools()
news_tool = DuckDuckGoTools()

# Create the AI Investment Agent
agent = Agent(
    name="AI Investment Agent",
    llm=llm,
    storage=storage,
    tools=[stock_tool, news_tool],
    description="An AI agent that provides stock market insights, sentiment analysis, and investment recommendations."
)

# Define how the agent processes user queries
def investment_assistant(query):
    if "stock price" in query.lower():
        return stock_tool.run(query)
    elif "news" in query.lower():
        return news_tool.run(query)
    else:
        return agent.chat(query)

# Start Agno Playground for testing
playground = Playground(agent)
serve_playground_app(playground, host="localhost", port=7777)

