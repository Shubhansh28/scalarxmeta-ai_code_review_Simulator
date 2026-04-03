import gradio as gr
from env.environment import CodeReviewEnv
from env.models import Action
from github_fetcher import fetch_full_pr
from ai_reviewer import analyze_pr
import json
import uuid
import os
from openai import OpenAI

# Global state
env = None
live_pr_data = None
live_ai_result = None


# ============================================================
# 🤖 ONE-CLICK AI AGENT for OpenEnv Simulator
# ============================================================

def run_ai_agent(task_type, task_index):
    """
    One-click: Loads a task, runs the AI agent step by step, returns the full
    results. The user just picks difficulty and clicks one button.
    """
    # 1. Reset
    sim_env = CodeReviewEnv(task_type=task_type, task_index=int(task_index))
    obs = sim_env.state()

    # Show the PR
    pr_display = f"## 📋 {obs.title}\n{obs.description}\n\n"
    for f in obs.files_changed:
        pr_display += f"### 📄 {f.filename}\n```diff\n{f.diff}\n```\n\n"

    # 2. Run AI
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN", "")

    if not HF_TOKEN:
        log = "❌ **HF_TOKEN not set.** Add it as a Secret in your HF Space settings."
        return pr_display, log, "0.00"

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    files_context = ""
    for f in obs.files_changed:
        files_context += f"--- {f.filename} ---\n{f.diff}\n\n"

    prompt = f"""You are a Senior Software Engineer reviewing a Pull Request.

### PR Details
- Title: {obs.title}
- Description: {obs.description}

### Code Changes
{files_context}

### Instructions
1. Find ALL bugs in the code. Be specific about file and line.
2. Each comment must be at least 10 words with a clear explanation.
3. After identifying bugs, decide: "approve" (no bugs) or "request_changes" (has bugs).

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
  "final_decision": "approve" | "request_changes",
  "decision_reason": "Why you approve or reject."
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a code reviewer. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500
        )
        ai_plan = json.loads(response.choices[0].message.content)
    except Exception as e:
        log = f"❌ AI Error: {str(e)}"
        return pr_display, log, "0.00"

    # 3. Execute each AI step against the environment
    log = "## 🤖 AI Agent Execution Log\n\n"
    step_num = 0

    for step in ai_plan.get("steps", []):
        action = Action(
            action_type=step.get("action_type", "comment"),
            comment=step.get("comment", ""),
            file=step.get("file"),
            line=step.get("line")
        )
        obs_new, reward, done, info = sim_env.step(action)
        step_num += 1

        icon = "✅" if reward > 0 else ("⚠️" if reward == 0 else "❌")
        log += f"### Step {step_num} {icon}\n"
        log += f"**Action:** `{step.get('action_type')}` on `{step.get('file', 'N/A')}`\n\n"
        log += f"**Comment:** {step.get('comment', '')}\n\n"
        log += f"**Reward:** `{reward:+.2f}` | **Running Score:** `{info.score:.2f}`\n\n---\n\n"

        if done:
            break

    # 4. Final decision
    if not done:
        final = ai_plan.get("final_decision", "approve")
        reason = ai_plan.get("decision_reason", "")
        final_action = Action(action_type=final, comment=reason)
        obs_new, reward, done, info = sim_env.step(final_action)
        step_num += 1

        icon = "✅" if reward > 0 else "❌"
        log += f"### Step {step_num} — Final Decision {icon}\n"
        log += f"**Action:** `{final.upper()}`\n\n"
        log += f"**Reason:** {reason}\n\n"
        log += f"**Reward:** `{reward:+.2f}` | **Final Score:** `{info.score:.2f}`\n\n"

    final_score = f"{info.score:.2f}"
    verdict = "✅ PASSED" if info.score >= 0.5 else "❌ FAILED"
    log += f"\n## 🏆 Final Result: {verdict} — Score: **{final_score}/1.00**\n"

    return pr_display, log, final_score


# ============================================================
# 🌐 LIVE PR REVIEW — Real-World GitHub Integration
# ============================================================

def fetch_live_pr(pr_url):
    global live_pr_data, live_ai_result
    live_ai_result = None

    if not pr_url or "github.com" not in pr_url or "/pull/" not in pr_url:
        return (
            "❌ **Invalid URL.** Please enter a valid GitHub PR link.\n\nExample: `https://github.com/owner/repo/pull/1`",
            "", "", ""
        )

    live_pr_data = fetch_full_pr(pr_url)

    if "error" in live_pr_data:
        return (f"❌ **Error:** {live_pr_data['error']}", "", "", "")

    meta = live_pr_data["metadata"]
    files = live_pr_data["files"]

    pr_info = f"""## 📋 {meta['title']}
**Author:** `{meta['author']}` • **Branch:** `{meta['head_branch']}` → `{meta['base_branch']}`
**Status:** `{meta['state']}` • **Mergeable:** `{meta['mergeable_state']}`
**Stats:** +{meta['additions']} additions, -{meta['deletions']} deletions across {meta['changed_files']} file(s)

---
**Description:** {meta['description']}
"""

    diff_display = ""
    for f in files:
        badge = "🟢" if f["status"] == "added" else ("🔴" if f["status"] == "removed" else "🟡")
        diff_display += f"### {badge} {f['filename']} ({f['status']})\n"
        diff_display += f"*+{f['additions']} / -{f['deletions']}*\n"
        diff_display += f"```diff\n{f['patch']}\n```\n\n"

    merge_status = ""
    if meta["mergeable_state"] == "clean":
        merge_status = "✅ **No merge conflicts detected.** This PR can be merged cleanly."
    elif meta["mergeable"] is False:
        merge_status = "⚠️ **Merge conflicts detected!** Resolve before merging."
    else:
        merge_status = f"ℹ️ Merge status: `{meta['mergeable_state']}`"

    return (
        pr_info, diff_display, merge_status,
        f"✅ Fetched {len(files)} file(s). Click **🤖 Run AI Review** to analyze."
    )


def run_ai_review():
    global live_pr_data, live_ai_result

    if live_pr_data is None or "error" in live_pr_data:
        return "❌ No PR data loaded. Fetch a PR first.", ""

    live_ai_result = analyze_pr(live_pr_data)

    comments_display = "## 🤖 AI Review Results\n\n"
    for i, comment in enumerate(live_ai_result.get("comments", []), 1):
        severity = comment.get("severity", "info")
        icon = "🔴" if severity == "error" else ("🟡" if severity == "warning" else "🔵")
        comments_display += f"### {icon} Finding #{i} — `{comment['file']}`\n"
        comments_display += f"**Severity:** {severity.upper()}\n\n"
        comments_display += f"{comment['comment']}\n\n---\n\n"

    verdict = live_ai_result.get("overall_verdict", "unknown")
    verdict_reason = live_ai_result.get("verdict_reason", "")
    verdict_icon = "✅" if verdict == "approve" else "❌"

    verdict_display = f"""## {verdict_icon} AI Verdict: **{verdict.upper()}**

{verdict_reason}

> **Note:** This is the AI's recommendation. The final decision is yours.
"""

    return comments_display, verdict_display


def user_approve():
    if live_pr_data is None:
        return "No PR loaded."
    meta = live_pr_data["metadata"]
    return f"""## ✅ PR APPROVED

**PR:** {meta['title']} by `{meta['author']}`

You can now merge on GitHub: 👉 [{meta['html_url']}]({meta['html_url']})
"""


def user_reject():
    if live_pr_data is None:
        return "No PR loaded."
    meta = live_pr_data["metadata"]
    findings = ""
    if live_ai_result and "comments" in live_ai_result:
        for c in live_ai_result["comments"]:
            if c["severity"] in ["error", "warning"]:
                findings += f"- **{c['file']}**: {c['comment']}\n"

    return f"""## ❌ PR REJECTED

**PR:** {meta['title']} by `{meta['author']}`

### Issues:
{findings if findings else "- Rejected based on your own judgment."}

Share feedback: 👉 [{meta['html_url']}]({meta['html_url']})
"""


# ============================================================
# 🎨 GRADIO UI
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="ScalarX Meta — AI Code Review") as demo:

    with gr.Tabs():

        # ====== TAB 1: Live PR Review ======
        with gr.TabItem("🌐 Live PR Review"):
            gr.Markdown("# 🌐 Live GitHub PR Review")
            gr.Markdown("Paste a GitHub PR URL → Fetch → AI analyzes → You decide.")

            with gr.Row():
                pr_url_input = gr.Textbox(
                    label="GitHub PR URL",
                    placeholder="https://github.com/owner/repo/pull/1",
                    scale=4
                )
                fetch_btn = gr.Button("📥 Fetch PR", variant="primary", scale=1)

            with gr.Row():
                with gr.Column(scale=1):
                    live_pr_info = gr.Markdown("Enter a GitHub PR URL and click **Fetch PR**.")
                    live_merge_status = gr.Markdown("")
                    live_status = gr.Markdown("")
                    gr.Markdown("---")
                    analyze_btn = gr.Button("🤖 Run AI Review", variant="secondary", size="lg")
                    gr.Markdown("---")
                    gr.Markdown("### 🧑‍💻 Your Decision")
                    with gr.Row():
                        approve_btn = gr.Button("✅ Approve PR", variant="primary")
                        reject_btn = gr.Button("❌ Reject PR", variant="stop")
                    user_decision_output = gr.Markdown("")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("📄 Code Changes"):
                            live_diff_view = gr.Markdown("Diff appears here after fetching.")
                        with gr.TabItem("🤖 AI Analysis"):
                            ai_comments_output = gr.Markdown("Click **Run AI Review** after fetching.")
                            ai_verdict_output = gr.Markdown("")

            fetch_btn.click(fetch_live_pr, inputs=[pr_url_input],
                            outputs=[live_pr_info, live_diff_view, live_merge_status, live_status])
            analyze_btn.click(run_ai_review, inputs=[], outputs=[ai_comments_output, ai_verdict_output])
            approve_btn.click(user_approve, inputs=[], outputs=[user_decision_output])
            reject_btn.click(user_reject, inputs=[], outputs=[user_decision_output])

        # ====== TAB 2: OpenEnv Simulator (SIMPLIFIED) ======
        with gr.TabItem("🛡️ OpenEnv Simulator"):
            gr.Markdown("# 🛡️ AI Code Review Benchmark")
            gr.Markdown("Pick a difficulty → Click **Run** → Watch the AI agent find bugs in real time.")

            with gr.Row():
                with gr.Column(scale=1):
                    sim_task_type = gr.Dropdown(
                        label="Difficulty",
                        choices=["syntax_review", "bug_detection", "full_review", "adversarial_review"],
                        value="syntax_review"
                    )
                    sim_task_index = gr.Number(label="Task Index", value=0, precision=0)
                    sim_run_btn = gr.Button("🤖 Run AI Agent", variant="primary", size="lg")
                    gr.Markdown("---")
                    sim_score = gr.Label(label="Final Score / 1.0")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("📋 Challenge"):
                            sim_pr_display = gr.Markdown("Select a difficulty and click **Run AI Agent**.")
                        with gr.TabItem("📊 AI Agent Log"):
                            sim_log = gr.Markdown("Results will appear here after the AI runs.")

            sim_run_btn.click(
                run_ai_agent,
                inputs=[sim_task_type, sim_task_index],
                outputs=[sim_pr_display, sim_log, sim_score]
            )

if __name__ == "__main__":
    demo.launch()
