import os
from dotenv import load_dotenv
from langchain_core.messages.ai import AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("GROQ_API_KEY")

# Instantiate the Groq LLM with the API key
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

# Define the tools (search, etc.)
tools = [TavilySearchResults(max_results=2)]

# Create the agent with the model and tools
agent = create_react_agent(
    model=groq_llm,
    tools=tools,
)


# Define a function to get the response
def get_response_from_ai_agents(llm_id, allow_search, query, provider):
    if provider == "Groq":
        # You could change model dynamically if needed
        groq_llm = ChatGroq(model=llm_id, api_key=api_key)
        tools = [TavilySearchResults(max_results=2)] if allow_search else []

        agent = create_react_agent(
            model=groq_llm,
            tools=tools,
        )

        # Pass messages in expected format
        state = {"messages": [query]}

        # Call the agent
        response = agent.invoke(state)

        # Extract AI messages
        messages = response.get("messages", [])
        ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]

        # Return the last AI message
        return ai_messages[-1] if ai_messages else "No AI response found."




