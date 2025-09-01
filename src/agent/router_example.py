import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import getpass
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
)


# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="The next step in the routing process"
    ) # type: ignore


# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)

# State
class State(TypedDict):
    input: str
    decision: str
    output: str


# Nodes
def story_node(state: State):
    """Write a story"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def joke_node(state: State):
    """Write a joke"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def poem_node(state: State):
    """Write a poem"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to story, joke, or poem based on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step} # type: ignore


# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "story":
        return "story_node"
    elif state["decision"] == "joke":
        return "joke_node"
    elif state["decision"] == "poem":
        return "poem_node"


# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("story_node", story_node)
router_builder.add_node("joke_node", joke_node)
router_builder.add_node("poem_node", poem_node)
router_builder.add_node("llm_call_router", llm_call_router)

# Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "story_node": "story_node",
        "joke_node": "joke_node",
        "poem_node": "poem_node",
    },
)
router_builder.add_edge("story_node", END)
router_builder.add_edge("joke_node", END)
router_builder.add_edge("poem_node", END)

# Compile workflow
router_workflow = router_builder.compile()

# input: {"input": "Write me a joke about cats"}