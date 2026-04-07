#!/bin/bash

# Configuration
# export HF_TOKEN="your_token_here"
export API_URL="${API_URL:-http://127.0.0.1:7860}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-32B-Instruct}"

echo "=== Starting Reproducibility Verification ==="
echo "Running 3 sequential evaluations..."
echo ""

for i in {1..3}
do
    echo "--- Run $i/3 ---"
    python3 inference.py | grep "\[SUMMARY\]"
    echo ""
done

echo "=== Verification Complete ==="
