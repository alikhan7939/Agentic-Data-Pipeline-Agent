"""State passed between LangGraph nodes."""
from typing import Dict, Any, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    dataset_path: str
    target_col: str
    messages: Annotated[list, add_messages]
    tool_results: Dict[str, Any]
    decisions_log: List[str]
    n_tool_calls: int
    final_report: Dict[str, Any]

