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

from utils.lm_utils import get_llm
llm = get_llm()

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


def chatbot(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process messages and return chatbot response.
    
    Uses Gemini to generate responses to user messages.
    """
    
    # Get response from Gemini
    response = llm.invoke(state["messages"])
    
    return {"messages": [response]}


# Define the graph
graph_builder = StateGraph(State, context_schema=Context)

# Add the chatbot node
graph_builder.add_node("chatbot", chatbot)

# Define the flow: START -> chatbot -> END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile(name="Chatbot")
