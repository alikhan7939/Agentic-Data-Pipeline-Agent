 # Agentic Data Pipeline Agent

An agentic layer on top of a ZenML pipeline. An LLM-based agent (LangGraph)
decides which pipeline steps to run — data quality checks, feature
engineering, outlier detection, model evaluation — and produces a final
report explaining both its decisions and the technical results.

## Structure

```
src/
├── pipelines/
│   └── data_pipeline.py     # ZenML pipeline wiring the steps together
├── steps/
│   ├── data_quality.py      # missing values / distribution report
│   ├── feature_engineering.py
│   ├── outlier_detection.py
│   ├── model_evaluation.py
│   └── reporting.py
├── agent/
│   ├── state.py             # Pydantic state passed between agent nodes
│   ├── tools.py             # LangGraph tools wrapping the ZenML steps
│   ├── prompts.py           # system prompt for the orchestrator agent
│   └── graph.py             # LangGraph graph definition (the agent itself)
└── api/
    └── main.py               # FastAPI endpoint to trigger a run
```

## How it fits together

1. `steps/*.py` — plain, testable Python functions. Each one is also
   decorated as a ZenML `@step` so it can run standalone inside a normal
   ZenML pipeline (`pipelines/data_pipeline.py`) if you don't want the
   agent in the loop.
2. `agent/tools.py` — thin wrappers that expose each step as a callable
   tool the LLM can invoke (same underlying function, tool-calling
   interface on top).
3. `agent/graph.py` — a LangGraph state machine: the LLM node decides
   which tool to call next based on results so far, loops until it thinks
   the pipeline is complete, then calls the reporting tool.
4. `api/main.py` — `POST /run` triggers a full agent run over a given
   dataset path and returns the final report.

## Setup

```bash
python -m venv zenml-env
source zenml-env/bin/activate   # WSL2 / Linux
pip install -r requirements.txt
cp .env.example .env            # add your ANTHROPIC_API_KEY
zenml init                      # first time only
```

## Run

Plain ZenML pipeline (no agent, all steps run in fixed order):
```bash
python -m src.pipelines.data_pipeline
```

Agentic run (agent decides the step order):
```bash
uvicorn src.api.main:app --reload
# then: curl -X POST localhost:8000/run -d '{"dataset_path": "data/churn.csv"}'
```

## Next steps to build out

- Replace the placeholder logic in each `steps/*.py` with your real
  Customer Churn preprocessing/XGBoost code.
- Give the agent a real stopping condition (currently: max 6 tool calls
  or "report generated").
- Add a `retrain_model` tool once you want the agent to close the loop
  (drift detected → retrain → redeploy).
- Swap the in-memory `AgentState` for the ZenML Artifact Store if you
  want run history persisted.
