from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from datetime import datetime

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
    confidence_gate: Optional[Dict] = None  # {passed, score, cases_run}


# ── Flywheel Models ──────────────────────────────────────────

class FlywheelCase(BaseModel):
    """A simulation case in the flywheel library."""
    case_id: str
    pr_id: str
    title: str
    description: str
    files_changed: List[dict]
    ground_truth_bugs: List[dict]
    expected_action: str = "request_changes"
    source: Literal["seed", "live_confirmed"] = "seed"
    language: str = "python"
    framework: str = "general"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DeveloperSignal(BaseModel):
    """Feedback from a developer on an AI review finding."""
    session_id: str
    signal_type: Literal["confirm_bug", "dismiss", "approve", "reject"]
    bug_index: Optional[int] = None  # Which AI finding (0-indexed)
    comment: Optional[str] = None


class ConfidenceAnnotation(BaseModel):
    """An AI review comment enriched with confidence data."""
    file: str
    severity: str
    comment: str
    confidence: float = 0.0           # 0-100 percentage
    confidence_source: str = "general_baseline"  # or "project_specific"
    is_novelty: bool = False


class PatternStats(BaseModel):
    """Historical accuracy for a specific bug pattern."""
    keyword: str
    times_flagged: int = 0
    times_confirmed: int = 0
    times_dismissed: int = 0
    decay_weight: float = 1.0  # Reduced on each dismissal
    accuracy: float = 0.0      # confirmed / flagged
