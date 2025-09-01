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

from utils.lm_utils import get_llm
llm = get_llm()

tool = TavilySearch(max_results=3)

base_system_prompt = "Be a good assistant. The user is called Francesco."

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

def get_api_key(runtime):
    if runtime.context and runtime.context.get('api_key'):
        return runtime.context.get('api_key')
    elif os.environ.get("GOOGLE_API_KEY"):
        return os.environ.get("GOOGLE_API_KEY")
    else:
        return None

def create_node(node_config: dict):
    """Create a node function with specific instructions from template."""
    
    def node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
        """Process messages and return chatbot response with custom instructions."""
        # Set up the API key from runtime context
        tools = [tool]
        llm_with_tools = llm.bind_tools(tools)
        
        # Prepare messages with system prompt
        messages_with_system = [
            {"role": "system", "content": base_system_prompt}
        ] + state["messages"] + [
            {"role": "user", "content": node_config['instructions']}
        ]
        print(messages_with_system)
        # Get response from Gemini
        response = llm_with_tools.invoke(messages_with_system)

        return {"messages": [response]}
    
    return node


template = [{
    "title": "Poem 1",
    "instructions": "write a poem about squirrels."
},
{
    "title": "Poem 2",
    "instructions": "ask the user what subject and then write a poem on that subject"
},
{
    "title": "Poem 3",
    "instructions": "write a poem about whales"
}]

graph_builder = StateGraph(State, context_schema=Context)

graph_builder.add_node("Poem 1", create_node(template[0]))
# graph_builder.add_node("Poem 2", create_node(template[1]))
# graph_builder.add_node("Poem 3", create_node(template[2]))

graph_builder.add_edge(START, "Poem 1")
# graph_builder.add_edge("Poem 1", "Poem 2")
# graph_builder.add_edge("Poem 2", "Poem 3")
graph_builder.add_edge("Poem 1", END)

graph = graph_builder.compile(name="Fran's poems")
