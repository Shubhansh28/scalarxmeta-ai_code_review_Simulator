"""
Unified Gradio UI — The Self-Learning Flywheel Dashboard.
Three tabs: Flywheel Review, OpenEnv Simulator, Flywheel Dashboard.
"""
import gradio as gr
from .environment import CodeReviewEnv
from .models import Action
from .github_fetcher import fetch_full_pr
from .ai_reviewer import analyze_pr
import json
import uuid
import os
from openai import OpenAI


# ============================================================
# 🔄 TAB 1: FLYWHEEL REVIEW (Merged Live PR + Confidence)
# ============================================================

def create_flywheel_review_handlers(store):
    """Create handler functions that close over the flywheel store."""

    # Session state (per-tab, not per-user — acceptable for demo)
    review_state = {
        "pr_data": None,
        "ai_result": None,
        "session_id": None,
        "benchmark_result": None,
    }

    def fetch_and_benchmark(pr_url):
        """Step 1+2: Fetch PR and run domain benchmark silently."""
        review_state["ai_result"] = None
        review_state["benchmark_result"] = None
        review_state["session_id"] = None

        if not pr_url or "github.com" not in pr_url or "/pull/" not in pr_url:
            return (
                "❌ **Invalid URL.** Please enter a valid GitHub PR link.\n\nExample: `https://github.com/owner/repo/pull/1`",
                "", "", "", ""
            )

        # Fetch PR
        pr_data = fetch_full_pr(pr_url)
        if "error" in pr_data:
            return (f"❌ **Error:** {pr_data['error']}", "", "", "", "")

        review_state["pr_data"] = pr_data
        review_state["session_id"] = str(uuid.uuid4())

        meta = pr_data["metadata"]
        files = pr_data["files"]

        # PR info panel
        pr_info = f"""## 📋 {meta['title']}
**Author:** `{meta['author']}` • **Branch:** `{meta['head_branch']}` → `{meta['base_branch']}`
**Status:** `{meta['state']}` • **Mergeable:** `{meta['mergeable_state']}`
**Stats:** +{meta['additions']} additions, -{meta['deletions']} deletions across {meta['changed_files']} file(s)

---
**Description:** {meta['description']}
"""

        # Diff display
        diff_display = ""
        for f in files:
            badge = "🟢" if f["status"] == "added" else ("🔴" if f["status"] == "removed" else "🟡")
            diff_display += f"### {badge} {f['filename']} ({f['status']})\n"
            diff_display += f"*+{f['additions']} / -{f['deletions']}*\n"
            diff_display += f"```diff\n{f['patch']}\n```\n\n"

        # Merge status
        if meta["mergeable_state"] == "clean":
            merge_status = "✅ **No merge conflicts detected.** This PR can be merged cleanly."
        elif meta["mergeable"] is False:
            merge_status = "⚠️ **Merge conflicts detected!** Resolve before merging."
        else:
            merge_status = f"ℹ️ Merge status: `{meta['mergeable_state']}`"

        # Step 2: Run domain benchmark
        from .confidence_engine import run_domain_benchmark
        benchmark = run_domain_benchmark(pr_data, store)
        review_state["benchmark_result"] = benchmark

        if benchmark["cases_run"] > 0:
            bench_icon = "✅" if benchmark["passed"] else "⚠️"
            bench_display = f"""### {bench_icon} Pre-Review Domain Benchmark

**Language:** `{benchmark['language']}` • **Framework:** `{benchmark['framework']}`
**Domain Score:** `{benchmark['score']:.0%}` (threshold: `{benchmark['threshold']:.0%}`)
**Cases Tested:** {benchmark['cases_run']}

{benchmark['message']}

"""
            for d in benchmark.get("details", []):
                src_badge = "🌱" if d.get("source") == "seed" else "🔵"
                bench_display += f"- {src_badge} `{d['case_id']}` — {d['title']} → Score: `{d['score']:.2f}`\n"
        else:
            bench_display = f"ℹ️ {benchmark.get('message', 'No benchmark data available.')}"

        status = f"✅ Fetched {len(files)} file(s). Click **🤖 Run AI Review** to analyze."

        return (pr_info, diff_display, merge_status, bench_display, status)


    def run_review():
        """Step 3+4: Run AI review with confidence annotations."""
        pr_data = review_state.get("pr_data")
        if pr_data is None or "error" in pr_data:
            return "❌ No PR data loaded. Fetch a PR first.", ""

        # Run raw AI analysis
        raw_result = analyze_pr(pr_data)

        # Annotate with confidence scores
        from .confidence_engine import annotate_comments
        annotated = annotate_comments(raw_result, store)
        review_state["ai_result"] = annotated

        # Register session in flywheel store for signal tracking
        session_id = review_state.get("session_id", str(uuid.uuid4()))
        store.register_review_session(session_id, pr_data, annotated)

        # Build rich display with confidence badges
        comments_display = "## 🤖 AI Review Results (Confidence-Annotated)\n\n"
        for i, comment in enumerate(annotated.get("comments", []), 0):
            severity = comment.get("severity", "info")
            icon = "🔴" if severity == "error" else ("🟡" if severity == "warning" else "🔵")

            # Confidence badge
            conf = comment.get("confidence", 50)
            conf_src = comment.get("confidence_source", "general_baseline")
            is_novelty = comment.get("is_novelty", False)

            if is_novelty:
                conf_badge = "🔮 **Novelty Alert** — Unfamiliar pattern"
            elif conf >= 75:
                conf_badge = f"🟢 **{conf:.0f}% confident**"
            elif conf >= 50:
                conf_badge = f"🟡 **{conf:.0f}% confident**"
            else:
                conf_badge = f"🔴 **{conf:.0f}% confident**"

            src_label = "general baseline" if conf_src == "general_baseline" else "project-specific"

            comments_display += f"### {icon} Finding #{i+1} — `{comment['file']}`\n"
            comments_display += f"**Severity:** {severity.upper()} • {conf_badge} ({src_label})\n\n"
            comments_display += f"{comment['comment']}\n\n"
            comments_display += f"*Pattern: `{comment.get('pattern_keyword', 'unknown')}`*\n\n---\n\n"

        # Verdict
        verdict = annotated.get("overall_verdict", "unknown")
        verdict_reason = annotated.get("verdict_reason", "")
        verdict_icon = "✅" if verdict == "approve" else "❌"

        verdict_display = f"""## {verdict_icon} AI Verdict: **{verdict.upper()}**

{verdict_reason}

> **Note:** This is the AI's recommendation. Use the buttons below to provide your feedback on each finding.
"""
        return comments_display, verdict_display


    def confirm_bug(bug_index_str):
        """Developer confirms a specific AI finding as a real bug."""
        session_id = review_state.get("session_id")
        if not session_id:
            return "❌ No active review session."

        try:
            bug_index = int(bug_index_str)
        except (ValueError, TypeError):
            return "❌ Enter a valid finding number (0-indexed)."

        from .feedback_bridge import capture_developer_signal
        result = capture_developer_signal(
            store=store,
            session_id=session_id,
            signal_type="confirm_bug",
            bug_index=bug_index,
        )

        msg = f"✅ **Finding #{bug_index + 1} confirmed as a real bug.**"
        if result.get("converted"):
            msg += f"\n\n🔄 **Flywheel activated!** A new simulation case `{result.get('case_id')}` has been added to the library."
        return msg


    def dismiss_bug(bug_index_str):
        """Developer dismisses a specific AI finding as a false positive."""
        session_id = review_state.get("session_id")
        if not session_id:
            return "❌ No active review session."

        try:
            bug_index = int(bug_index_str)
        except (ValueError, TypeError):
            return "❌ Enter a valid finding number (0-indexed)."

        from .feedback_bridge import capture_developer_signal
        result = capture_developer_signal(
            store=store,
            session_id=session_id,
            signal_type="dismiss",
            bug_index=bug_index,
        )

        return f"❌ **Finding #{bug_index + 1} dismissed as false positive.** Pattern weight reduced."


    def approve_pr():
        pr_data = review_state.get("pr_data")
        if not pr_data:
            return "No PR loaded."
        meta = pr_data["metadata"]
        return f"""## ✅ PR APPROVED

**PR:** {meta['title']} by `{meta['author']}`

You can now merge on GitHub: 👉 [{meta['html_url']}]({meta['html_url']})
"""

    def reject_pr():
        pr_data = review_state.get("pr_data")
        if not pr_data:
            return "No PR loaded."
        meta = pr_data["metadata"]
        ai_result = review_state.get("ai_result")
        findings = ""
        if ai_result and "comments" in ai_result:
            for c in ai_result["comments"]:
                if c["severity"] in ["error", "warning"]:
                    findings += f"- **{c['file']}**: {c['comment']}\n"
        return f"""## ❌ PR REJECTED

**PR:** {meta['title']} by `{meta['author']}`

### Issues:
{findings if findings else "- Rejected based on your own judgment."}

Share feedback: 👉 [{meta['html_url']}]({meta['html_url']})
"""

    return {
        "fetch_and_benchmark": fetch_and_benchmark,
        "run_review": run_review,
        "confirm_bug": confirm_bug,
        "dismiss_bug": dismiss_bug,
        "approve_pr": approve_pr,
        "reject_pr": reject_pr,
    }


# ============================================================
# 📊 TAB 3: FLYWHEEL DASHBOARD
# ============================================================

def get_dashboard_data(store):
    """Generate dashboard markdown from store stats."""
    stats = store.get_library_stats()
    patterns = store.get_all_pattern_stats()

    # Header stats
    dashboard = f"""## 📊 Flywheel Health

| Metric | Value |
|:---|:---|
| **Total Simulation Cases** | {stats['total_cases']} |
| **Seed Cases** | {stats['seed_cases']} |
| **Live-Generated Cases** | {stats['live_cases']} |
| **Patterns Tracked** | {stats['total_patterns_tracked']} |

### 🌍 Cases by Language
"""
    for lang, count in stats.get("by_language", {}).items():
        dashboard += f"- **{lang.capitalize()}**: {count} cases\n"

    # Recent cases
    dashboard += "\n### 🕐 Recent Cases\n"
    if stats.get("recent_cases"):
        dashboard += "| Case ID | Title | Source |\n|:---|:---|:---|\n"
        for c in stats["recent_cases"]:
            src_badge = "🌱 Seed" if c["source"] == "seed" else "🔵 Live"
            dashboard += f"| `{c['case_id']}` | {c['title']} | {src_badge} |\n"
    else:
        dashboard += "*No cases yet.*\n"

    # Pattern accuracy
    dashboard += "\n### 🎯 Pattern Accuracy\n"
    if patterns:
        dashboard += "| Pattern | Flagged | Confirmed | Dismissed | Accuracy | Weight |\n"
        dashboard += "|:---|:---:|:---:|:---:|:---:|:---:|\n"
        for kw, p in patterns.items():
            dashboard += f"| `{kw}` | {p['times_flagged']} | {p['times_confirmed']} | {p['times_dismissed']} | {p.get('accuracy', 0):.0f}% | {p['decay_weight']:.2f} |\n"
    else:
        dashboard += "*No patterns tracked yet. Review some PRs to start the flywheel!*\n"

    return dashboard


# ============================================================
# 🎨 MAIN UI ASSEMBLY
# ============================================================

def create_demo(store):
    """Build the full Gradio UI with flywheel store injected."""

    handlers = create_flywheel_review_handlers(store)

    with gr.Blocks(theme=gr.themes.Soft(), title="ScalarX Meta — Self-Learning Flywheel") as demo:

        gr.Markdown("# 🔄 ScalarX Meta — Self-Learning Flywheel")
        gr.Markdown("*The more you review, the smarter it gets.*")

        with gr.Tabs():

            # ====== TAB 1: Flywheel Review ======
            with gr.TabItem("🔄 Flywheel Review"):
                gr.Markdown("# 🔄 Live PR Review with Confidence Scoring")
                gr.Markdown("Paste a GitHub PR URL → Fetch & Benchmark → AI analyzes with confidence → You confirm or dismiss each finding → Flywheel learns.")

                with gr.Row():
                    pr_url_input = gr.Textbox(
                        label="GitHub PR URL",
                        placeholder="https://github.com/owner/repo/pull/1",
                        scale=4
                    )
                    fetch_btn = gr.Button("📥 Fetch & Benchmark", variant="primary", scale=1)

                with gr.Row():
                    with gr.Column(scale=1):
                        live_pr_info = gr.Markdown("Enter a GitHub PR URL and click **Fetch & Benchmark**.")
                        live_merge_status = gr.Markdown("")
                        live_benchmark_status = gr.Markdown("")
                        live_status = gr.Markdown("")
                        gr.Markdown("---")
                        analyze_btn = gr.Button("🤖 Run AI Review", variant="secondary", size="lg")
                        gr.Markdown("---")

                        # Confirm / Dismiss individual findings
                        gr.Markdown("### 🔬 Per-Finding Feedback")
                        gr.Markdown("Enter the finding number (starting from 0) to confirm or dismiss:")
                        with gr.Row():
                            bug_index_input = gr.Number(label="Finding #", value=0, precision=0, scale=1)
                        with gr.Row():
                            confirm_btn = gr.Button("✅ Confirm Bug", variant="primary", scale=1)
                            dismiss_btn = gr.Button("❌ Not a Bug", variant="stop", scale=1)
                        signal_output = gr.Markdown("")

                        gr.Markdown("---")
                        gr.Markdown("### 🧑‍💻 Final Decision")
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

                # Wire up events
                fetch_btn.click(
                    handlers["fetch_and_benchmark"],
                    inputs=[pr_url_input],
                    outputs=[live_pr_info, live_diff_view, live_merge_status, live_benchmark_status, live_status]
                )
                analyze_btn.click(
                    handlers["run_review"],
                    inputs=[],
                    outputs=[ai_comments_output, ai_verdict_output]
                )
                confirm_btn.click(
                    handlers["confirm_bug"],
                    inputs=[bug_index_input],
                    outputs=[signal_output]
                )
                dismiss_btn.click(
                    handlers["dismiss_bug"],
                    inputs=[bug_index_input],
                    outputs=[signal_output]
                )
                approve_btn.click(handlers["approve_pr"], inputs=[], outputs=[user_decision_output])
                reject_btn.click(handlers["reject_pr"], inputs=[], outputs=[user_decision_output])


            # ====== TAB 2: Flywheel Dashboard ======
            with gr.TabItem("📊 Flywheel Dashboard"):
                gr.Markdown("# 📊 Self-Learning Flywheel Dashboard")
                gr.Markdown("Track the health of your simulation library and pattern accuracy.")

                dashboard_display = gr.Markdown(get_dashboard_data(store))
                dashboard_refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")

                dashboard_refresh_btn.click(
                    lambda: get_dashboard_data(store),
                    inputs=[],
                    outputs=[dashboard_display]
                )

    return demo
