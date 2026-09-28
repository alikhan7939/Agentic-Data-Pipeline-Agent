"""The agent itself: a small LangGraph state machine.

llm_node -> decides next tool call (or stops)
tool_node -> executes the chosen tool, appends result to state
report_node -> once the LLM stops calling tools, build the final report
"""
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS
from src.agent.prompts import SYSTEM_PROMPT
from src.steps.reporting import run_reporting

load_dotenv()

MAX_TOOL_CALLS = 6

llm = ChatGroq(
    model=os.environ.get("GROQ_MODEL"),
    api_key=os.environ.get("GROQ_API_KEY"),
).bind_tools(ALL_TOOLS)


def llm_node(state: AgentState) -> dict:
    messages = state["messages"]
    if not messages:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Dataset: {state['dataset_path']}, "
                    f"target column: {state['target_col']}. "
                    "Decide which tool to run first."
                )
            ),
        ]
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if state["n_tool_calls"] >= MAX_TOOL_CALLS:
        return "report"
    if getattr(last, "tool_calls", None):
        return "tools"
    return "report"


def record_tool_results(state: AgentState) -> dict:
    """Runs after ToolNode: pull latest tool results into tool_results /
    decisions_log so the final report has a clean structure."""
    tool_results = dict(state.get("tool_results", {}))
    decisions_log = list(state.get("decisions_log", []))

    for msg in state["messages"][-1:]:
        name = getattr(msg, "name", None)
        if name:
            tool_results[name] = msg.content
            decisions_log.append(f"Ran `{name}`")

    return {
        "tool_results": tool_results,
        "decisions_log": decisions_log,
        "n_tool_calls": state["n_tool_calls"] + 1,
    }


def report_node(state: AgentState) -> dict:
    report = run_reporting(
        tool_results=state.get("tool_results", {}),
        decisions_log=state.get("decisions_log", []),
    )
    return {"final_report": report}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("record", record_tool_results)
    graph.add_node("report", report_node)

    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "report": "report"})
    graph.add_edge("tools", "record")
    graph.add_edge("record", "llm")
    graph.add_edge("report", END)

    return graph.compile()


def run_agent(dataset_path: str, target_col: str = "Churn") -> dict:
    app = build_graph()
    initial_state: AgentState = {
        "dataset_path": dataset_path,
        "target_col": target_col,
        "messages": [],
        "tool_results": {},
        "decisions_log": [],
        "n_tool_calls": 0,
        "final_report": {},
    }
    final_state = app.invoke(initial_state)
    return final_state["final_report"]
