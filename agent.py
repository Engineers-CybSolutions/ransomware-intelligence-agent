
import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

llm = ChatOpenAI(
    api_key=os.environ["HARNESS_INFERENCE_API_KEY"],
    base_url=os.environ["HARNESS_INFERENCE_BASE_URL"],
    model=os.environ["HARNESS_INFERENCE_MODEL"],
    temperature=0
)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def respond(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}

workflow = StateGraph(AgentState)
workflow.add_node("respond", respond)
workflow.add_edge(START, "respond")
workflow.add_edge("respond", END)

agent = workflow.compile()
