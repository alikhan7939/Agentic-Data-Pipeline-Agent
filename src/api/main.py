from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from src.agent.graph import run_agent
import os

load_dotenv(override=True)

app = FastAPI(title="Agentic Data Pipeline Agent")


class RunRequest(BaseModel):
    dataset_path: str
    target_col: str = "churn"


@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/run")
def run(request: RunRequest):
    report = run_agent(request.dataset_path, request.target_col)
    return report


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug-env")
def debug_env():
    key = os.environ.get("GROQ_API_KEY")
    return {
        "key_prefix": key[:10] if key else None,
        "key_len": len(key) if key else 0,
        "model": os.environ.get("GROQ_MODEL"),
        "cwd": os.getcwd(),
    }