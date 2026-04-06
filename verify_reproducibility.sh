#!/bin/bash

# Configuration
# export HF_TOKEN="your_token_here"
export API_URL="http://localhost:7860"
export MODEL_NAME="google/gemma-2-9b-it:free"

echo "=== Starting Reproducibility Verification ==="
echo "Running 3 sequential evaluations..."
echo ""

for i in {1..3}
do
    echo "--- Run $i/3 ---"
    python3 oracle_inference.py | grep "\[SUMMARY\]"
    echo ""
done

echo "=== Verification Complete ==="
