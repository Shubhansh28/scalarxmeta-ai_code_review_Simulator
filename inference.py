import os
import json
import asyncio
import httpx
from typing import List, Optional, Set, Dict, Any
from openai import OpenAI

# Required Environment Variables (Mandatory for Hackathon)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", "your_token_here"))

# Environment Endpoint (Defaults to local for validation, override with API_URL)
API_URL = os.getenv("API_URL", "http://localhost:7860")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Format action to be concise for logs
    action_log = action.replace("\n", " ").strip()[:100]
    print(f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def get_model_message(
    client: OpenAI, 
    step: int, 
    obs: Dict[str, Any], 
    last_reward: float, 
    history: List[Dict[str, Any]],
    found_issues: Set[str]
) -> Dict[str, Any]:
    """
    Generates the next action using the LLM.
    """
    prompt = f"""
    Step {step}: Last Reward: {last_reward}
    
    ### Task Context
    - PR Title: {obs.get('title')}
    - Description: {obs.get('description')}
    - Files Changed: {json.dumps(obs.get('files_changed'), indent=2)}
    
    ### Review History
    {json.dumps(history[-3:], indent=2)}
    
    ### Instructions
    1. Identify bugs and use 'comment' to point them out. 
    2. Once done, 'request_changes' (if bugs found) or 'approve' (if none).
    3. Output JSON.
    
    {{
      "action_type": "comment" | "approve" | "request_changes",
      "file": "filename",
      "line": line_number,
      "comment": "Analysis of the code."
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a senior engineer auditing code. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"action_type": "comment", "file": "unknown", "line": 0, "comment": f"Analysis pending... ({e})"}

async def run_baseline_task(task_type: str, task_index: int = 0) -> float:
    """
    Runs a single task through the environment.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    BENCHMARK = "code_review_env"
    TASK_NAME = f"{task_type}_{task_index}"
    MAX_STEPS = 8
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    history: List[Dict[str, Any]] = []
    found_issues: Set[str] = set()
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
            
            last_reward = 0.0
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done: break

                # 2. Get Model Action
                action_dict = await get_model_message(client, step, obs, last_reward, history, found_issues)

                # 3. Step Environment
                resp = await http_client.post(f"{API_URL}/step", json={
                    "session_id": session_id,
                    "action": {
                        "action_type": action_dict.get("action_type", "comment"),
                        "file": action_dict.get("file"),
                        "line": action_dict.get("line"),
                        "comment": action_dict.get("comment")
                    }
                })
                step_result = resp.json()
                
                obs = step_result["observation"]
                reward = step_result["reward"]
                done = step_result["done"]
                info = step_result["info"]
                
                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                
                if action_dict.get("action_type") == "comment":
                    found_issues.add(f"{action_dict.get('file')}:{action_dict.get('line')}")

                log_step(step=step, action=action_dict.get("action_type", "unknown"), reward=reward, done=done, error=None)
                history.append({"step": step, "action": action_dict.get("action_type"), "reward": reward})

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
    tasks = [
        ("syntax_review", 0),
        ("bug_detection", 0),
        ("adversarial_review", 0)
    ]
    scores = []
    for t_type, t_idx in tasks:
        scores.append(await run_baseline_task(t_type, t_idx))
    
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n[SUMMARY] Evaluation Complete. Avg Score: {avg:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
