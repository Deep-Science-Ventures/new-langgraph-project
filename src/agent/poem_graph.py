"""LangGraph chatbot implementation.

A simple chatbot that uses Gemini to respond to user messages.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Dict, TypedDict

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

tool = TavilySearch(max_results=3)

custom_system_prompt = "Be a good assistant. The user is called Francesco."

class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    api_key: str


class State(TypedDict):
    """State for the chatbot agent.

    Messages have the type "list". The `add_messages` function
    in the annotation defines how this state key should be updated
    (in this case, it appends messages to the list, rather than overwriting them)
    """
    
    messages: Annotated[list, add_messages]


def address_the_user(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process messages and return chatbot response.
    
    Uses Gemini to generate responses to user messages.
    """
    # Set up the API key from runtime context
    api_key = None
    if runtime.context and runtime.context.get('api_key'):
        api_key = runtime.context.get('api_key')
    elif os.environ.get("GOOGLE_API_KEY"):
        api_key = os.environ.get("GOOGLE_API_KEY")
    else:
        return {
            "messages": [{
                "role": "assistant",
                "content": "Error: GOOGLE_API_KEY not provided in context or environment variables."
            }]
        }
    
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key
    )
    tools = [tool]
    llm_with_tools = llm.bind_tools(tools)

    # Prepare messages with system prompt
    messages_with_system = [
        {"role": "system", "content": custom_system_prompt}
    ] + state["messages"]
    
    # Get response from Gemini
    response = llm_with_tools.invoke(messages_with_system)

    return {"messages": [response]}


graph_builder = StateGraph(State, context_schema=Context)

graph_builder.add_node("address_the_user", address_the_user)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "address_the_user",
    tools_condition,
)
graph_builder.add_edge("tools", "address_the_user")
graph_builder.add_edge(START, "address_the_user")

graph = graph_builder.compile(name="Chatbot with Web Search")
