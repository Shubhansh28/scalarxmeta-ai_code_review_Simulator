from typing import Any, Dict, Tuple, Optional
from .models import Observation, Action, Reward, Info
from .tasks import get_task

class CodeReviewEnv:
    def __init__(self, task_type: str = "syntax_review", task_index: int = 0, max_steps: int = 8, custom_data: dict = None, flywheel_store=None):
        self.task_type = task_type
        self.task_index = task_index
        self.max_steps = max_steps
        self.custom_data = custom_data
        self.flywheel_store = flywheel_store  # Optional store reference for benchmark tracking
        self.reset()
        
    def reset(self) -> Observation:
        if self.task_type == "custom" and self.custom_data:
            self.task_data = self.custom_data
        else:
            self.task_data = get_task(self.task_type, self.task_index)
            
        self.step_count = 0
        self.comments_history = []
        self.actions_history = []
        self.bugs_identified = set()
        self.done = False
        self.total_score = 0.0
        self.last_action_feedback = "Environment initialized. Awaiting review."
        return self.state()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Info]:
        if self.done:
            return self.state(), 0.0, True, Info(done=True, score=self.total_score, message="Episode already finished")

        self.step_count += 1
        self.actions_history.append(action.dict())
        
        from .graders import evaluate_step
        step_reward, new_bugs = evaluate_step(self.task_data, action, self.bugs_identified)
        self.bugs_identified.update(new_bugs)
        
        step_reward -= 0.05 # Step penalty
        
        if action.action_type == "comment":
            self.comments_history.append(f"[{action.file}:{action.line}] {action.comment}")
            self.last_action_feedback = f"Added comment on {action.file}."
        elif action.action_type in ["approve", "request_changes"]:
            self.done = True
            self.last_action_feedback = f"PR finished with action: {action.action_type}."
            
        if self.step_count >= self.max_steps:
            self.done = True
            self.last_action_feedback = "Max steps reached."

        self.total_score += step_reward
        self.total_score = max(0.0, self.total_score)

        if self.done:
            from .graders import finalize_episode
            penalty = finalize_episode(self.task_data, self.bugs_identified)
            self.total_score += penalty
            
            # Final clamping for the official 0.0-1.0 range
            self.total_score = max(0.0, min(1.0, self.total_score))
            info = Info(done=True, score=self.total_score, message="Episode completed")

            # Record benchmark result to flywheel store if available
            self._record_benchmark_result()
        else:
            info = Info(done=False, score=self.total_score)

        return self.state(), step_reward, self.done, info

    def _record_benchmark_result(self):
        """Record episode results to the flywheel store for pattern tracking."""
        if self.flywheel_store is None:
            return
        try:
            ground_truth = self.task_data.get("ground_truth_bugs", [])
            for i, bug in enumerate(ground_truth):
                keyword = bug.get("keyword", "")
                if keyword:
                    if i in self.bugs_identified:
                        self.flywheel_store.record_confirmation(keyword)
                    else:
                        self.flywheel_store.record_flag(keyword)
        except Exception:
            pass  # Never crash the environment over analytics

    def state(self) -> Observation:
        return Observation(
            pr_id=self.task_data.get("pr_id", ""),
            title=self.task_data.get("title", ""),
            description=self.task_data.get("description", ""),
            files_changed=self.task_data.get("files_changed", []),
            comments_history=self.comments_history,
            step_count=self.step_count,
            max_steps=self.max_steps,
            last_action_feedback=self.last_action_feedback
        )
