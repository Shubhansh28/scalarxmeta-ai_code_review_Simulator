import os
import json
import asyncio
import httpx
from typing import List, Optional, Set, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Required Environment Variables (Mandatory for Hackathon)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
# Prioritize HF_TOKEN as per requirements, fallback to API_KEY or OPENAI_API_KEY
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

# Environment Endpoint
API_URL = os.getenv("API_URL", "http://localhost:7860")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Serialize action to a single line for the log_step requirement
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

async def get_model_message(
    client: OpenAI, 
    step: int, 
    obs: Dict[str, Any], 
    last_reward: float, 
    history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generates the next action using the model."""
    prompt = f"""
    Step {step}: Last Reward: {last_reward}
    Title: {obs.get('title')}
    Description: {obs.get('description')}
    Diff: {json.dumps(obs.get('files_changed'), indent=1)}
    History: {json.dumps(history[-2:], indent=1)}
    
    Output JSON ONLY:
    {{
      "reasoning": "thought process",
      ### Instructions
      1. CRITICAL: You must use the 'comment' action to point out EVERY bug you find with line-level detail.
      2. You will be PENALIZED if you use 'request_changes' without having made a 'comment' first.
      3. Only after you have commented on all bugs, provide a final decision ('request_changes' or 'approve').
      4. Output JSON ONLY.
      "action_type": "comment" | "approve" | "request_changes",
      "file": "filename",
      "line": 42,
      "comment": "10+ word review comment"
    }}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a senior engineer auditing code. Precision is key. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0  # Force deterministic output
        )
        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            return {
              "action_type": "comment",
              "file": "unknown",
              "line": 0,
              "comment": "Parsing error fallback"
            }
    except Exception as e:
        return {"action_type": "comment", "file": "unknown", "line": 0, "comment": f"Continuing analysis... (Error: {e})"}

async def run_baseline_task(task_type: str, task_index: int = 0) -> float:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    BENCHMARK = "code_review_env"
    TASK_NAME = f"{task_type}_{task_index}"
    MAX_STEPS = 8
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    history: List[Dict[str, Any]] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    async with httpx.AsyncClient(timeout=20.0) as http_client:
        try:
            # OpenENV.reset()
            resp = await http_client.post(f"{API_URL}/reset", json={
                "task_type": task_type,
                "task_index": task_index
            })
            result = resp.json()
            session_id = result["session_id"]
            obs = result["observation"]
            
            last_reward = 0.0
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done: break

                action_dict = await get_model_message(client, step, obs, last_reward, history)
                
                # Check for repetition
                action_type = action_dict.get("action_type", "comment")
                if action_type not in ["comment", "approve", "request_changes"]:
                    action_type = "comment"
                    action_dict["action_type"] = action_type
                    
                action_str = f"{action_type}: {action_dict.get('comment', 'no_comment')[:60]}...".replace("\n", " ")

                # Step Environment
                resp = await http_client.post(f"{API_URL}/step", json={
                    "session_id": session_id,
                    "action": action_dict
                })
                step_result = resp.json()
                
                obs = step_result["observation"]
                reward = step_result["reward"]
                done = step_result["done"]
                info = step_result.get("info", {})
                
                # Stop if repeating the same bug (score reduction prevention)
                if reward < 0 and action_type == "comment":
                    # If we got a penalty for a comment, we should stop and decide.
                    # We'll take one more step to finalize.
                    final_action = "request_changes" if sum(rewards) > 0.3 else "approve"
                    resp = await http_client.post(f"{API_URL}/step", json={
                        "session_id": session_id,
                        "action": {"action_type": final_action, "comment": "Finalizing review based on findings."}
                    })
                    step_result = resp.json()
                    done = True
                    info = step_result.get("info", {})
                
                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                
                log_step(step=step, action=action_str, reward=reward, done=done, error=None)
                history.append({"step": step, "type": action_type, "reward": reward})

                if done:
                    score = info.get("score", 0.0)
                    break

            success = score >= 0.5
        except Exception as e:
            log_step(step=steps_taken+1, action="error", reward=0.0, done=True, error=str(e))
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

async def main() -> None:
    tasks = [("syntax_review", 0), ("bug_detection", 0), ("full_review", 0), ("adversarial_review", 0)]
    total_score = 0.0
    for t_type, t_idx in tasks:
        total_score += await run_baseline_task(t_type, t_idx)
    print(f"\n[SUMMARY] Avg Score: {total_score/len(tasks):.3f}")

if __name__ == "__main__":
    asyncio.run(main())
