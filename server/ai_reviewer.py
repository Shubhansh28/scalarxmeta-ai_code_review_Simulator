"""
AI PR Analyzer — Sends real GitHub PR diffs to Qwen2.5-Coder for analysis.
Returns structured, file-level review comments (min 10 words each).
No dummy data. No hallucination. Real diffs from real PRs.
"""
import os
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "your_huggingface_token_here")


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
        diff_context += f"\n--- File: {f['filename']} ({f['status']}) ---\n"
        diff_context += f"  Additions: {f['additions']}, Deletions: {f['deletions']}\n"
        diff_context += f"```diff\n{f['patch']}\n```\n"

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

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a production-grade code reviewer. Output ONLY valid JSON. Never hallucinate issues that don't exist in the diff. Be precise."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {
            "comments": [{"file": "system", "severity": "error", "comment": f"AI analysis failed: {str(e)}. Please check your HF_TOKEN."}],
            "overall_verdict": "request_changes",
            "verdict_reason": f"Analysis could not be completed due to an error: {str(e)}",
            "merge_conflicts_found": False
        }
