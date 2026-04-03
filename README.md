---
title: Scalarxmeta
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🚀 ScalarX Meta: AI Code Review Simulator

![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green?style=for-the-badge)
![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow?style=for-the-badge)

**ScalarX Meta** is a production-grade, OpenEnv-compliant simulation environment designed to benchmark and improve the logical reasoning and security-auditing capabilities of AI agents. It models the high-stakes cognitive task of Pull Request (PR) review, challenging agents to identify subtle bugs and architectural flaws across multi-file diffs.

---

## 🌟 Key Features

*   **Real-World Workflow**: Simulates the daily life of a Senior Software Engineer (inspecting diffs, commenting, and making final decisions).
*   **Adversarial Focus**: Includes specialized tasks where code *looks* correct but contains hidden logic holes or security vulnerabilities.
*   **Hacker-Proof Graders**: A deterministic reward system that penalizes "keyword dumping" and rewards deep, qualitative explanations (min 10 words).
*   **Custom Task Upload**: Explicit support for uploading and testing your own private PRs and codebases directly through the UI or API.
*   **Dual-Mode Interface**: 
    *   **API Mode**: FastAPI endpoints (`/reset`, `/step`) for automated RL agent training and evaluation.
    *   **Visual Mode**: A beautiful Gradio dashboard for manual human auditing and debugging.

---

## 🛠 Technical Specification

### 🕹️ Action Space
The agent interacts using a discrete-choice action with structured metadata:
*   `comment`: Provide inline feedback on a specific file/line. (Requires **10+ words** for full reward).
*   `approve`: Finalize the review and accept the PR.
*   `request_changes`: Reject the PR due to identified defects. (Requires justification via previous comments).

### 📊 Observation Space
Information provided to the agent at each step:
*   **Context**: PR Title and Description.
*   **Codebase**: Filenames and Unified Diffs of all changes.
*   **Thread**: History of existing comments and previous action rewards.
*   **State**: Step count and remaining budget.

---

## 📈 Evaluation Tasks & Difficulty

| Task ID | Difficulty | Focus |
| :--- | :--- | :--- |
| `syntax_review` | **Easy** | Syntax errors, naming conventions, and mutable default arguments. |
| `bug_detection` | **Medium** | Logical errors, incorrect loop boundaries, and off-by-one bugs. |
| `full_review` | **Hard** | Complex multi-file dependencies and architectural regressions. |
| `adversarial` | **Expert** | **Deceptive code** designed to bypass basic static analysis. |

---

## 🚀 Getting Started

### 1. Local Development
```bash
# Install dependencies
pip install -e .

# Start the API & Visual Dashboard
uvicorn app:app --port 7860
```
Visit `http://localhost:7860` to access the interactive Gradio UI.

### 2. Testing Custom Code
You can now upload your own PRs for testing:
1.  Navigate to the **🛠️ Custom PR Creator** tab in the UI.
2.  Paste your PR Diffs and define the **Bug Metadata** (for automated grading).
3.  Click **🚀 Load Custom Challenge** to start the evaluation.

### 3. Running the AI Baseline
Ensure your environment variables are configured:
```bash
export HF_TOKEN="your_huggingface_token"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
python3 inference.py
```

---

## 🐳 Deployment (Hugging Face Spaces)

This project is fully containerized and compatible with Hugging Face Spaces.

1.  Create a new Space with the **Docker** SDK.
2.  Add your `HF_TOKEN` and `MODEL_NAME` as **Secrets** in the Space settings.
3.  The agent will automatically be reachable at `https://your-space-name.hf.space`.

---

## 🏆 Model Benchmarks (Baseline)

| Model | Avg. Score | Logic | Security | Robustness |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-4o** | **0.925** | 0.95 | 1.00 | 0.85 |
| **Claude 3.5 Sonnet** | **0.905** | 0.92 | 1.00 | 0.82 |
| **Qwen 2.5 Coder 32B** | **0.880** | 0.90 | 0.95 | 0.80 |
| **Llama 3.1 70B** | **0.750** | 0.80 | 0.85 | 0.55 |

---
**OpenEnv Compliant** • **State-of-the-art Evaluation** • **Built for Hackathons**
