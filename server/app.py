from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, Dict, List
import uuid
import time
import gradio as gr

from .environment import CodeReviewEnv
from .models import Action, Observation, Reward, DeveloperSignal
from .flywheel_store import FlywheelStore
from .feedback_bridge import capture_developer_signal
from .confidence_engine import run_domain_benchmark, annotate_comments
from .gradio_ui import create_demo

app = FastAPI(title="ScalarX Meta — Self-Learning Flywheel")

# Shared state
# Sessions stored as {session_id: (env, created_at_timestamp)}
SESSION_TTL_SECONDS = 1800  # 30 minutes — prevents memory leaks on 8GB machines
sessions: Dict[str, tuple] = {}
flywheel_store = FlywheelStore()


def _cleanup_stale_sessions():
    """Remove sessions older than SESSION_TTL_SECONDS."""
    now = time.time()
    stale = [sid for sid, (_, ts) in sessions.items() if now - ts > SESSION_TTL_SECONDS]
    for sid in stale:
        del sessions[sid]


# ── Existing Environment Endpoints ───────────────────────────

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
def reset_env(req: Optional[ResetRequest] = Body(None)):
    _cleanup_stale_sessions()  # Prevent memory leaks
    if req is None:
        req = ResetRequest()
        
    env = CodeReviewEnv(
        task_type=req.task_type,
        task_index=req.task_index,
        max_steps=req.max_steps,
        flywheel_store=flywheel_store,
    )
    session_id = str(uuid.uuid4())
    sessions[session_id] = (env, time.time())
    
    obs = env.state()
    return {
        "session_id": session_id,
        "observation": obs.dict()
    }

@app.post("/reset/custom")
def reset_env_custom(req: CustomResetRequest):
    _cleanup_stale_sessions()
    custom_data = req.dict()
    env = CodeReviewEnv(
        task_type="custom",
        custom_data=custom_data,
        max_steps=req.max_steps,
        flywheel_store=flywheel_store,
    )
    session_id = str(uuid.uuid4())
    sessions[session_id] = (env, time.time())
    
    obs = env.state()
    return {
        "session_id": session_id,
        "observation": obs.dict()
    }

@app.post("/step")
def step_env(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env, _ = sessions[req.session_id]
    obs, reward, done, info = env.step(req.action)
    
    # Clamp reward to [0.0, 1.0] per OpenEnv spec before returning
    reward_clamped = float(max(0.0, min(1.0, reward)))
    
    return {
        "observation": obs.dict(),
        "reward": reward_clamped,
        "done": done,
        "info": info.dict()
    }

class StateRequest(BaseModel):
    session_id: str

@app.get("/state")
def get_state(session_id: str):
    """GET /state?session_id=... for quick status checks."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    env, _ = sessions[session_id]
    return {"observation": env.state().dict()}

@app.post("/state")
def post_state(req: StateRequest):
    """POST /state per OpenEnv interface spec."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    env, _ = sessions[req.session_id]
    return {"observation": env.state().dict()}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "scalarx_meta_flywheel"}


# ── Flywheel API Endpoints ───────────────────────────────────

@app.post("/flywheel/signal")
def flywheel_signal(signal: DeveloperSignal):
    """Receive developer feedback on a live review finding."""
    result = capture_developer_signal(
        store=flywheel_store,
        session_id=signal.session_id,
        signal_type=signal.signal_type,
        bug_index=signal.bug_index,
        comment=signal.comment,
    )
    return result

@app.get("/flywheel/stats")
def flywheel_stats():
    """Return flywheel library statistics."""
    return flywheel_store.get_library_stats()

@app.get("/flywheel/cases")
def flywheel_cases():
    """List all simulation cases in the flywheel library."""
    cases = flywheel_store.get_all_cases()
    return {
        "total": len(cases),
        "cases": [
            {
                "case_id": c.get("case_id"),
                "title": c.get("title"),
                "source": c.get("source"),
                "language": c.get("language"),
                "created_at": c.get("created_at"),
            }
            for c in cases
        ]
    }

@app.get("/flywheel/patterns")
def flywheel_patterns():
    """Return all pattern accuracy statistics."""
    return flywheel_store.get_all_pattern_stats()

@app.post("/flywheel/export")
def flywheel_export():
    """Export the full flywheel store for backup."""
    return flywheel_store.export_data()

@app.post("/flywheel/import")
def flywheel_import(data: dict = Body(...)):
    """Import a previously exported flywheel store."""
    flywheel_store.import_data(data)
    return {"status": "imported", "cases": len(flywheel_store.cases)}


# ── Mount Gradio UI ──────────────────────────────────────────

demo = create_demo(flywheel_store)
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
