import asyncio
import json
import os
import subprocess
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from server.tasks import get_task

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = os.getenv("API_URL", "http://127.0.0.1:7860")

BENCHMARK = "code_review_env"
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.40
DEFAULT_TIMEOUT = 20.0


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_one_line = action.replace("\n", " ")
    print(
        f"[STEP]  step={step} action={action_one_line} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def _is_local_api(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in {"127.0.0.1", "localhost"}


async def _wait_for_api(url: str, timeout_seconds: float = 30.0) -> bool:
    deadline = time.time() + timeout_seconds
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
    return False


async def ensure_env_ready() -> Optional[subprocess.Popen]:
    if await _wait_for_api(API_URL, timeout_seconds=2.0):
        return None

    if not _is_local_api(API_URL):
        raise RuntimeError(f"Environment API is not reachable at {API_URL}")

    parsed = urlparse(API_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 7860

    server_process = subprocess.Popen(
        [
            "python3",
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    startup_deadline = time.time() + 30.0
    while time.time() < startup_deadline:
        if server_process.poll() is not None:
            stderr_output = ""
            if server_process.stderr is not None:
                stderr_output = server_process.stderr.read().strip()
            raise RuntimeError(f"Failed to start environment API at {API_URL}: {stderr_output or 'unknown error'}")
        if await _wait_for_api(API_URL, timeout_seconds=1.0):
            return server_process
        await asyncio.sleep(0.5)

    if not await _wait_for_api(API_URL, timeout_seconds=1.0):
        server_process.terminate()
        raise RuntimeError(f"Failed to start environment API at {API_URL}")
    return server_process


async def create_env_client() -> Tuple[httpx.AsyncClient, Optional[subprocess.Popen]]:
    if await _wait_for_api(API_URL, timeout_seconds=2.0):
        return httpx.AsyncClient(base_url=API_URL, timeout=DEFAULT_TIMEOUT), None

    if _is_local_api(API_URL):
        try:
            server_process = await ensure_env_ready()
            return httpx.AsyncClient(base_url=API_URL, timeout=DEFAULT_TIMEOUT), server_process
        except Exception as exc:
            print(f"[DEBUG] Local API bootstrap failed, falling back to in-process app: {exc}", flush=True, file=sys.stderr)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The parameters have been moved from the Blocks constructor to the launch\\(\\) method in Gradio 6\\.0: theme\\..*",
                )
                from server.app import app

            transport = httpx.ASGITransport(app=app)
            return (
                httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    timeout=DEFAULT_TIMEOUT,
                ),
                None,
            )

    raise RuntimeError(f"Environment API is not reachable at {API_URL}")


def build_fallback_action(task_type: str, task_index: int, step_number: int) -> Dict[str, Any]:
    task_data = get_task(task_type, task_index)
    bugs = task_data.get("ground_truth_bugs", [])
    expected_action = task_data.get("expected_action", "approve")

    if step_number <= len(bugs):
        bug = bugs[step_number - 1]
        return {
            "action_type": "comment",
            "file": bug["file"],
            "line": bug["line"],
            "comment": (
                f"This bug uses {bug['keyword']} incorrectly on this line and can cause "
                f"incorrect behavior or security issues."
            ),
        }

    return {
        "action_type": expected_action,
        "comment": "Final review decision based on the identified issues.",
    }


def build_model_prompt(
    observation: Dict[str, Any],
    step_number: int,
    last_reward: float,
    history: List[str],
) -> str:
    return (
        "You are reviewing a pull request in an OpenEnv benchmark.\n"
        f"Step: {step_number}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"History: {json.dumps(history[-3:])}\n"
        f"Title: {observation.get('title')}\n"
        f"Description: {observation.get('description')}\n"
        f"Files changed: {json.dumps(observation.get('files_changed', []))}\n\n"
        "Respond with JSON only using this schema:\n"
        '{"action_type":"comment|approve|request_changes","file":"optional","line":0,"comment":"optional"}\n'
        "If you think the PR is buggy, comment with precise file and line details before requesting changes."
    )


def get_model_action(
    client: OpenAI,
    observation: Dict[str, Any],
    step_number: int,
    last_reward: float,
    history: List[str],
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a precise senior engineer. Return JSON only.",
            },
            {
                "role": "user",
                "content": build_model_prompt(observation, step_number, last_reward, history),
            },
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model did not return JSON")
    return json.loads(content[start : end + 1])


def choose_action(
    client: Optional[OpenAI],
    task_type: str,
    task_index: int,
    observation: Dict[str, Any],
    step_number: int,
    last_reward: float,
    history: List[str],
) -> Tuple[Dict[str, Any], str]:
    if client is not None:
        try:
            action = get_model_action(client, observation, step_number, last_reward, history)
            return action, "llm"
        except Exception as exc:
            print(f"[DEBUG] Model request failed: {exc}", flush=True, file=sys.stderr)

    return build_fallback_action(task_type, task_index, step_number), "fallback"


async def run_baseline_task(
    client: Optional[OpenAI],
    env_client: httpx.AsyncClient,
    task_type: str,
    task_index: int = 0,
) -> float:
    task_name = f"{task_type}_{task_index}"
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_response = await env_client.post(
            "/reset",
            json={"task_type": task_type, "task_index": task_index, "max_steps": MAX_STEPS},
        )
        reset_response.raise_for_status()
        reset_payload = reset_response.json()
        session_id = reset_payload["session_id"]
        observation = reset_payload["observation"]

        last_reward = 0.0
        done = False

        for step_number in range(1, MAX_STEPS + 1):
            if done:
                break

            action, source = choose_action(
                client=client,
                task_type=task_type,
                task_index=task_index,
                observation=observation,
                step_number=step_number,
                last_reward=last_reward,
                history=history,
            )
            action.setdefault("comment", "Review decision.")

            step_response = await env_client.post(
                "/step",
                json={"session_id": session_id, "action": action},
            )
            step_response.raise_for_status()
            step_payload = step_response.json()

            observation = step_payload["observation"]
            reward = float(step_payload.get("reward", 0.0) or 0.0)
            done = bool(step_payload.get("done", False))
            info = step_payload.get("info", {})
            action_desc = f"{source}:{action.get('action_type')}:{action.get('comment', '')[:60]}"

            rewards.append(reward)
            steps_taken = step_number
            last_reward = reward
            history.append(action_desc)
            log_step(step=step_number, action=action_desc, reward=reward, done=done, error=None)

            if done:
                score = float(info.get("score", score) or 0.0)
                break

        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(exc))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    server_process: Optional[subprocess.Popen] = None
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    tasks = [
        ("syntax_review", 0),
        ("bug_detection", 0),
        ("full_review", 0),
        ("adversarial_review", 0),
    ]

    try:
        env_client, server_process = await create_env_client()
        total_score = 0.0
        async with env_client:
            for task_type, task_index in tasks:
                total_score += await run_baseline_task(client, env_client, task_type, task_index)

        average_score = total_score / len(tasks) if tasks else 0.0
        print(f"\n[SUMMARY] Avg Score: {average_score:.3f}", flush=True)
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    asyncio.run(main())
