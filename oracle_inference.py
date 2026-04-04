import os
import json
import asyncio
import httpx
from typing import List, Optional, Set, Dict, Any

# Environment Endpoint
API_URL = os.getenv("API_URL", "http://localhost:7860")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run_oracle_task(task_type: str, task_index: int = 0) -> float:
    BENCHMARK = "code_review_env"
    TASK_NAME = f"{task_type}_{task_index}"
    MAX_STEPS = 8
    
    log_start(task=TASK_NAME, env=BENCHMARK, model="ORACLE_DETERMINISTIC")
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    async with httpx.AsyncClient() as http_client:
        try:
            # 1. Reset Environment
            resp = await http_client.post(f"{API_URL}/reset", json={
                "task_type": task_type,
                "task_index": task_index
            })
            result = resp.json()
            session_id = result["session_id"]
            obs = result["observation"]
            
            # Extract ground truth from the environment (Oracle hack for testing only)
            # In a real scenario, the agent wouldn't have this.
            from server.tasks import get_task
            task_data = get_task(task_type, task_index)
            bugs = task_data.get("ground_truth_bugs", [])
            expected_action = task_data.get("expected_action", "approve")

            # 2. Sequential Discovery
            for i, bug in enumerate(bugs):
                action_payload = {
                    "action_type": "comment",
                    "file": bug["file"],
                    "line": 42, # Mock line
                    "comment": f"Found a bug: {bug['keyword']} handles this incorrectly."
                }
                
                resp = await http_client.post(f"{API_URL}/step", json={
                    "session_id": session_id,
                    "action": action_payload
                })
                step_result = resp.json()
                reward = step_result["reward"]
                rewards.append(reward)
                log_step(step=i+1, action=f"comment_{bug['keyword']}", reward=reward, done=False, error=None)

            # 3. Final Decision
            final_resp = await http_client.post(f"{API_URL}/step", json={
                "session_id": session_id,
                "action": {"action_type": expected_action, "comment": "Final review decision."}
            })
            final_data = final_resp.json()
            score = final_data["info"].get("score", 0.0)
            rewards.append(final_data["reward"])
            log_step(step=len(bugs)+1, action=expected_action, reward=final_data["reward"], done=True, error=None)

            success = score >= 0.5
            steps_taken = len(bugs) + 1

        except Exception as e:
            log_step(step=0, action="error", reward=0.0, done=True, error=str(e))
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score

async def main() -> None:
    tasks = [("syntax_review", 0), ("bug_detection", 0), ("adversarial_review", 0)]
    total_score = 0.0
    for t_type, t_idx in tasks:
        total_score += await run_oracle_task(t_type, t_idx)
    print(f"\n[SUMMARY] Avg Score: {total_score/len(tasks):.3f}")

if __name__ == "__main__":
    asyncio.run(main())
