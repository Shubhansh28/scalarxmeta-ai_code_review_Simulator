"""
AI PR Analyzer — Sends real GitHub PR diffs to Qwen2.5-Coder for analysis.
Returns structured, file-level review comments (min 10 words each).
No dummy data. No hallucination. Real diffs from real PRs.
"""
import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai")
HF_TOKEN = os.getenv("HF_TOKEN", "any_string_for_pollinations")

if not HF_TOKEN or HF_TOKEN == "your_huggingface_token_here":
    raise ValueError("HF_TOKEN or OPENROUTER_API_KEY environment variable is missing or invalid. Please check your .env file.")


def analyze_pr(pr_data: dict) -> list:
    """
    Takes the output of github_fetcher.fetch_full_pr() and sends each
    file's patch to the AI model for deep code review.
    
    Returns a list of structured review comments:
    [
        {
            "file": "style.css",
            "severity": "warning" | "error" | "info",
            "comment": "10+ word explanation of the issue found."
        },
        ...
    ]
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    metadata = pr_data["metadata"]
    files = pr_data["files"]
    
    # Build the real diff context for the model
    diff_context = ""
    for f in files:
        patch = f.get("patch", "")
        if len(patch) > 5000:
            patch = patch[:5000] + "\n... (diff truncated for length) ..."
        diff_context += f"\n--- File: {f['filename']} ({f['status']}) ---\n"
        diff_context += f"  Additions: {f['additions']}, Deletions: {f['deletions']}\n"
        diff_context += f"```diff\n{patch}\n```\n"

    prompt = f"""You are a Senior Software Engineer performing a thorough code review.

### Pull Request Metadata (REAL DATA — do NOT hallucinate or make up information)
- **Title**: {metadata['title']}
- **Author**: {metadata['author']}
- **Description**: {metadata['description']}
- **Branch**: {metadata['head_branch']} → {metadata['base_branch']}
- **Merge Status**: {metadata['mergeable_state']}
- **Stats**: +{metadata['additions']} additions, -{metadata['deletions']} deletions across {metadata['changed_files']} file(s)

### Code Changes (Unified Diff)
{diff_context}

### Your Instructions
1. Review EVERY file change carefully for:
   - Logical errors, incorrect CSS/JS behavior
   - Accessibility issues (missing focus states, contrast, etc.)
   - Responsiveness problems
   - Security concerns
   - Best practice violations
   - Merge conflict markers (<<<<<<, ======, >>>>>>)
2. Each comment MUST be at least 10 words long with a clear explanation.
3. Do NOT hallucinate issues. If the code looks correct, say so.
4. Use severity levels: "error" (bugs/breaking), "warning" (potential issues), "info" (suggestions).
5. At the end, give an overall_verdict: "approve" or "request_changes" with a reason.

Output ONLY this JSON structure:
{{
  "comments": [
    {{
      "file": "filename",
      "severity": "error" | "warning" | "info",
      "comment": "Detailed explanation of the issue (minimum 10 words)."
    }}
  ],
  "overall_verdict": "approve" | "request_changes",
  "verdict_reason": "Summary of why you approve or reject this PR.",
  "merge_conflicts_found": false
}}
"""

    # Fallback models for Hugging Face Serverless tier
    models_to_try = [MODEL_NAME, "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]
    retries_per_model = 2
    last_error = None
    
    for model in models_to_try:
        for attempt in range(retries_per_model):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a code reviewer. Output ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2500,
                    timeout=120
                )
                
                raw_content = response.choices[0].message.content
                if not raw_content:
                    raise ValueError(f"Model {model} returned empty content")

                content = raw_content.strip()
                # Robust extraction of the first JSON object
                if "{" in content and "}" in content:
                    start_idx = content.find("{")
                    stack = 0
                    first_obj_end = -1
                    for i in range(start_idx, len(content)):
                        if content[i] == "{":
                            stack += 1
                        elif content[i] == "}":
                            stack -= 1
                            if stack == 0:
                                first_obj_end = i
                                break
                    if first_obj_end != -1:
                        content = content[start_idx:first_obj_end+1]

                result = json.loads(content)
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"AI analysis with {model} (attempt {attempt+1}) failed: {e}")
                # Handle model loading (503)
                if "503" in str(e) or "loading" in str(e).lower():
                    import time
                    time.sleep(15) # Long wait for model load
                    continue
                if attempt < retries_per_model - 1:
                    import time
                    time.sleep(2)
                    continue
        
        logger.info(f"Model {model} failed, trying next fallback...")

    # If all fallbacks fail
    logger.error(f"AI analysis failed for all models: {last_error}")
    return {
        "comments": [{"file": "system", "severity": "error", "comment": f"AI analysis failed across multiple models ({str(last_error)}). Please try again in a few minutes."}],
        "overall_verdict": "request_changes",
        "verdict_reason": f"Analysis could not be completed after several attempts: {str(last_error)}",
        "merge_conflicts_found": False
    }
