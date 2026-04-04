from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
import gradio as gr

from .environment import CodeReviewEnv
from .models import Action, Observation
from .gradio_ui import demo

app = FastAPI(title="Code Review Environment")

# In-memory store for sessions
sessions: Dict[str, CodeReviewEnv] = {}

class ResetRequest(BaseModel):
    task_type: str = "syntax_review"
    task_index: int = 0
    max_steps: int = 8

class StepRequest(BaseModel):
    session_id: str
    action: Action

class CustomResetRequest(BaseModel):
    title: str
    description: str
    files_changed: list
    expected_bugs: list
    max_steps: int = 8

@app.post("/reset")
def reset_env(req: ResetRequest):
    env = CodeReviewEnv(
        task_type=req.task_type,
        task_index=req.task_index,
        max_steps=req.max_steps
    )
    session_id = str(uuid.uuid4())
    sessions[session_id] = env
    
    obs = env.state()
    return {
        "session_id": session_id,
        "observation": obs.dict()
    }

@app.post("/reset/custom")
def reset_env_custom(req: CustomResetRequest):
    custom_data = req.dict()
    env = CodeReviewEnv(
        task_type="custom",
        custom_data=custom_data,
        max_steps=req.max_steps
    )
    session_id = str(uuid.uuid4())
    sessions[session_id] = env
    
    obs = env.state()
    return {
        "session_id": session_id,
        "observation": obs.dict()
    }

@app.post("/step")
def step_env(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env = sessions[req.session_id]
    obs, reward, done, info = env.step(req.action)
    
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info.dict()
    }

@app.get("/state")
def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env = sessions[session_id]
    return {
        "observation": env.state().dict()
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "code_review_env"}

# Mount Gradio UI at the root (must be after all routes are defined)
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
