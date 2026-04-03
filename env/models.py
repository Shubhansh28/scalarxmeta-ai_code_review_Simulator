from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class FileChange(BaseModel):
    filename: str
    diff: str

class Observation(BaseModel):
    pr_id: str
    title: str
    description: str
    files_changed: List[FileChange]
    comments_history: List[str]
    step_count: int
    max_steps: int
    last_action_feedback: str

class Action(BaseModel):
    action_type: Literal["comment", "approve", "request_changes"]
    file: Optional[str] = None
    line: Optional[int] = None
    comment: Optional[str] = None

class Reward(BaseModel):
    reward: float

class Info(BaseModel):
    done: bool
    score: Optional[float] = None
    message: Optional[str] = None
