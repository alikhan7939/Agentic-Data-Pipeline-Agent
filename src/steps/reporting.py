"""Final report generation — combines every tool result the agent
collected along the way plus the agent's own reasoning trace.
"""
from typing import Dict, Any, List
from datetime import datetime, timezone

from zenml import step

 
def _build_report(
    tool_results: Dict[str, Any], decisions_log: List[str] = [0]
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "steps_run": list(tool_results.keys()),
        "results": tool_results,
        "agent_reasoning": decisions_log,
    }


@step
def reporting_step(data_quality, outlier_detection, model_metrics):
    tool_results = {
        "data_quality": data_quality,
        "outlier_detection": outlier_detection,
        "model_metrics": model_metrics,
    }
    return _build_report(tool_results,)
    
def run_reporting(
    tool_results: Dict[str, Any], decisions_log: List[str]
) -> Dict[str, Any]:
    return _build_report(tool_results, decisions_log)
