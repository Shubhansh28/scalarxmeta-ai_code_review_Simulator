---
title: ScalarX Meta — AI Code Review Simulator
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# 🚀 ScalarX Meta: AI Code Review Simulator

![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green?style=for-the-badge)
![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow?style=for-the-badge)

**ScalarX Meta** is a production-grade, OpenEnv-compliant simulation environment designed to benchmark and train AI agents in the high-stakes cognitive task of **Pull Request (PR) review**. It challenges agents to identify subtle bugs, architectural flaws, and security vulnerabilities across multi-file diffs.

---

## 🕹️ Action Space
The agent interacts using structured JSON actions via the `step()` API:
*   `comment`: Provide inline feedback on a specific `file` and `line`.
*   `approve`: Finalize the review and accept the PR.
*   `request_changes`: Reject the PR due to identified defects.

## 📊 Observation Space
At each step, the agent receives:
*   **PR Metadata**: Title, Description, and Author.
*   **Diff Context**: Unified diffs for all changed files.
*   **History**: A transcript of previous comments and rewards.
*   **Budget**: Current step count vs. `max_steps`.

---

## 📉 Self-Learning Flywheel
Unique to ScalarX Meta is a **Self-Learning Flywheel** (accessible via the Web UI). It tracks agent performance patterns and stores them in `flywheel_store.json`. This data is used to:
1.  Identify common agent failure modes in bug detection.
2.  Provide human-in-the-loop (HITL) feedback to refine the rewards.
3.  Generate better synthetic task data over time.

---

## 📈 Evaluation Tasks
| Task ID | Difficulty | Focus |
| :--- | :--- | :--- |
| `syntax_review` | **Easy** | Syntax errors, naming conventions, and basic best practices. |
| `bug_detection` | **Medium** | Logical errors, incorrect loop boundaries, and memory leaks. |
| `full_review` | **Hard** | Complex multi-file regressions and architectural flaws. |
| `adversarial_review` | **Expert** | **Deceptive code** designed to bypass basic static analysis. |

---

## 🚀 Getting Started (Hackathon Setup)

### 1. Requirements
Ensure you have the following environment variables defined:
*   `API_BASE_URL`: The LLM inference endpoint (e.g., Hugging Face Router or OpenRouter).
*   `MODEL_NAME`: The model identifier (e.g., `google/gemma-2-9b-it:free`).
*   `HF_TOKEN` or `OPENROUTER_API_KEY`: Your provider's API key.

### 2. Local Setup
```bash
# Clone and install dependencies
git clone <your-repo-url>
cd scalarxmeta
pip install -r requirements.txt

# Create a .env file with your credentials (HF or OpenRouter)
echo "OPENROUTER_API_KEY=your_token" > .env

# Start the environment server
uvicorn server.app:app --port 7860

# In a new terminal, run the baseline inference
python3 inference.py
```

### 3. Docker Deployment
```bash
docker build -t openenv-scalarx .
docker run -p 7860:7860 -e HF_TOKEN="your_token" openenv-scalarx
```

---

## 📝 Mandatory STDOUT Logging
The `inference.py` script emits structured logs strictly following the OpenEnv requirement (note usage of double-spaces after `[STEP]` for alignment parsing):

```text
[START] task=<task_name> env=code_review_env model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

---

## 🏆 Baseline Performance (Qwen 2.5 Coder 32B)
| Task | Success Rate | Avg. Score |
| :--- | :---: | :---: |
| Syntax Review | 95% | 0.92 |
| Bug Detection | 82% | 0.85 |
| Full Review | 65% | 0.68 |
| Adversarial | **45%** | **0.52** |

---
**Built for the OpenEnv Hackathon** • **Engineered for Precision** • **Deployable on HF Spaces**
