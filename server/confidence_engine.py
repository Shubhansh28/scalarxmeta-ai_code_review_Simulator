"""
Confidence Engine — Runs domain-matched simulations behind the scenes
and attaches confidence scores to live review comments.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ── Domain Benchmark ─────────────────────────────────────────

def run_domain_benchmark(
    pr_data: Dict[str, Any],
    store,
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Select domain-matched simulation cases and run the AI against them.
    Returns {passed: bool, score: float, cases_run: int, details: [...]}.
    
    This runs synchronously (blocking) since it's a quick pre-check.
    """
    from .feedback_bridge import detect_language, detect_framework
    from .environment import CodeReviewEnv
    from .models import Action

    files = pr_data.get("files", [])
    all_diffs = " ".join(f.get("patch", "") for f in files)
    language = detect_language(files)
    framework = detect_framework(files, all_diffs)

    # Get domain-matched cases
    cases = store.get_domain_cases(language=language, framework=framework, limit=3)

    if not cases:
        return {
            "passed": True,  # No cases = can't block, let it through
            "score": 0.0,
            "cases_run": 0,
            "language": language,
            "framework": framework,
            "details": [],
            "message": f"No simulation cases found for {language}/{framework}. Proceeding with general baseline."
        }

    # Run agent against each case
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN:
        return {
            "passed": True,
            "score": 0.0,
            "cases_run": 0,
            "language": language,
            "framework": framework,
            "details": [],
            "message": "HF_TOKEN not set. Skipping benchmark."
        }

    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    total_score = 0.0
    details = []

    for case in cases:
        try:
            env = CodeReviewEnv(
                task_type="custom",
                custom_data=case,
                max_steps=4,  # Quick benchmark, fewer steps
            )
            obs = env.state()

            # Build context for AI
            files_context = ""
            for f in obs.files_changed:
                files_context += f"--- {f.filename} ---\n{f.diff}\n\n"

            prompt = f"""You are benchmarking your code review skills. Find ALL bugs.

### PR Details
- Title: {obs.title}
- Description: {obs.description}

### Code Changes
{files_context}

Output ONLY this JSON:
{{
  "steps": [
    {{
      "action_type": "comment",
      "file": "filename",
      "line": 0,
      "comment": "10+ word explanation of the bug found."
    }}
  ],
  "final_decision": "approve" | "request_changes"
}}"""

            # Priority Tiers for Simulations
            hf_models = [MODEL_NAME, "meta-llama/Llama-3.1-8B-Instruct"]
            poll_models = ["openai", "qwen"]
            
            hf_client = client
            poll_client = OpenAI(base_url="https://text.pollinations.ai/openai", api_key="not-needed")
            
            ai_plan = None
            last_error = None
            retries_per_model = 2

            # Tier 1: Hugging Face (Best Quality)
            for model in hf_models:
                for attempt in range(retries_per_model):
                    try:
                        response = hf_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a code reviewer. Output ONLY valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            max_tokens=800,
                            timeout=120
                        )
                        
                        raw_content = response.choices[0].message.content
                        if not raw_content: raise ValueError("Empty")
                        
                        content = raw_content.strip()
                        if "{" in content and "}" in content:
                            start_idx = content.find("{")
                            stack = 0
                            end_idx = -1
                            for i in range(start_idx, len(content)):
                                if content[i] == "{": stack += 1
                                elif content[i] == "}":
                                    stack -= 1
                                    if stack == 0:
                                        end_idx = i
                                        break
                            if end_idx != -1:
                                ai_plan = json.loads(content[start_idx:end_idx+1])
                                break
                    except Exception as e:
                        last_error = e
                        if "402" in str(e) or "credits" in str(e).lower(): break # Skip to next tier
                if ai_plan: break

            # Tier 2 Fallback: Pollinations AI (Unlimited Backup)
            if not ai_plan:
                for model in poll_models:
                    try:
                        response = poll_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a code reviewer. Output ONLY valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            max_tokens=800,
                            timeout=60
                        )
                        raw_content = response.choices[0].message.content
                        if not raw_content: continue
                        
                        content = raw_content.strip()
                        if "{" in content and "}" in content:
                            start_idx = content.find("{")
                            stack = 0
                            end_idx = -1
                            for i in range(start_idx, len(content)):
                                if content[i] == "{": stack += 1
                                elif content[i] == "}":
                                    stack -= 1
                                    if stack == 0:
                                        end_idx = i
                                        break
                            if end_idx != -1:
                                ai_plan = json.loads(content[start_idx:end_idx+1])
                                break
                    except Exception as e:
                        last_error = e
                if ai_plan: break
            
            if not ai_plan:
                logger.error(f"Benchmark failed for all models for case {case.task}")
                details.append({"case": case.task, "score": 0.0, "error": "AI failed"})
                continue

            # Execute steps
            for step in ai_plan.get("steps", [])[:3]:
                action = Action(
                    action_type=step.get("action_type", "comment"),
                    comment=step.get("comment", ""),
                    file=step.get("file"),
                    line=step.get("line"),
                )
                _, _, done, info = env.step(action)
                if done:
                    break

            # Final decision
            if not env.done:
                final = ai_plan.get("final_decision", "approve")
                action = Action(action_type=final, comment="Benchmark decision.")
                _, _, _, info = env.step(action)

            case_score = info.score or 0.0
            total_score += case_score
            details.append({
                "case_id": case.get("case_id", "unknown"),
                "title": case.get("title", ""),
                "score": round(case_score, 2),
                "source": case.get("source", "seed"),
            })

        except Exception as e:
            logger.warning(f"Benchmark case {case.get('case_id')} failed: {e}")
            details.append({
                "case_id": case.get("case_id", "unknown"),
                "title": case.get("title", ""),
                "score": 0.0,
                "error": str(e),
            })

    avg_score = total_score / len(cases) if cases else 0.0
    passed = avg_score >= threshold

    return {
        "passed": passed,
        "score": round(avg_score, 2),
        "cases_run": len(cases),
        "language": language,
        "framework": framework,
        "threshold": threshold,
        "details": details,
        "message": (
            f"✅ AI passed domain benchmark ({avg_score:.0%} ≥ {threshold:.0%})"
            if passed else
            f"⚠️ AI below confidence threshold ({avg_score:.0%} < {threshold:.0%}). Human review recommended."
        )
    }


# ── Confidence Scoring ───────────────────────────────────────

def compute_confidence(
    comment_text: str,
    store,
) -> Dict[str, Any]:
    """
    Look up the AI's historical accuracy on patterns mentioned in this comment
    and compute a confidence score.
    """
    from .feedback_bridge import _extract_keyword

    keyword = _extract_keyword(comment_text)
    stats = store.get_pattern_stats(keyword)

    if stats and stats.get("times_flagged", 0) >= 3:
        # Enough data for project-specific confidence
        accuracy = stats.get("accuracy", 0.0)
        return {
            "confidence": round(accuracy, 1),
            "confidence_source": "project_specific",
            "keyword": keyword,
            "is_novelty": False,
        }
    elif stats and stats.get("times_flagged", 0) > 0:
        # Some data but not enough — treat as novelty
        return {
            "confidence": round(stats.get("accuracy", 50.0), 1),
            "confidence_source": "general_baseline",
            "keyword": keyword,
            "is_novelty": True,
        }
    else:
        # No data at all — novelty alert
        return {
            "confidence": 50.0,
            "confidence_source": "general_baseline",
            "keyword": keyword,
            "is_novelty": True,
        }


def annotate_comments(
    ai_result: Dict[str, Any],
    store,
) -> Dict[str, Any]:
    """
    Enrich each AI review comment with confidence data.
    Returns a new ai_result dict with annotated comments.
    """
    annotated = dict(ai_result)
    annotated_comments = []

    for comment in ai_result.get("comments", []):
        enriched = dict(comment)
        conf = compute_confidence(comment.get("comment", ""), store)
        enriched["confidence"] = conf["confidence"]
        enriched["confidence_source"] = conf["confidence_source"]
        enriched["is_novelty"] = conf["is_novelty"]
        enriched["pattern_keyword"] = conf["keyword"]

        # Record flag in store
        store.record_flag(conf["keyword"])

        annotated_comments.append(enriched)

    annotated["comments"] = annotated_comments
    return annotated


def classify_novelty(comment_text: str, store) -> bool:
    """Check if a pattern has < 3 simulation cases covering it."""
    from .feedback_bridge import _extract_keyword
    keyword = _extract_keyword(comment_text)
    stats = store.get_pattern_stats(keyword)
    if not stats or stats.get("times_flagged", 0) < 3:
        return True
    return False
