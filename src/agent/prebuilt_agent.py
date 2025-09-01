from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import getpass
from typing_extensions import Literal
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, ToolMessage
from agent.agent import add, divide, multiply
from utils.lm_utils import get_llm

llm = get_llm()
tools = [add, multiply, divide]

# Pass in:
# (1) the augmented LLM with tools
# (2) the tools list (which is used to create the tool node)
pre_built_agent = create_react_agent(llm, tools=tools)

# Show the agent
# display(Image(pre_built_agent.get_graph().draw_mermaid_png()))

# Invoke
# messages = [HumanMessage(content="Add 3 and 4.")]
# messages = pre_built_agent.invoke({"messages": messages},{"recursion_limit": recursion_limit})
# for m in messages["messages"]:
#     m.pretty_print()