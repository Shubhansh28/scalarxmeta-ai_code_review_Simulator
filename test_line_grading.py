import sys
import os
sys.path.append(os.getcwd())

from server.graders import evaluate_step
from server.models import Action

# Mock task data
task_data = {
    "ground_truth_bugs": [
        {"type": "syntax", "file": "profiles.py", "keyword": "Age", "line": 12}
    ],
    "expected_action": "request_changes"
}

def test_case(name, action_dict, expected_reward):
    action = Action(**action_dict)
    reward, new_bugs = evaluate_step(task_data, action, set())
    print(f"Test: {name}")
    print(f"  Result: {reward:.2f} (Expected: {expected_reward:.2f})")
    assert abs(reward - expected_reward) < 0.01

print("--- Starting Grader Precision Tests ---")

# 1. Perfect Match (Keyword + Line)
test_case("Perfect Match", 
          {"action_type": "comment", "file": "profiles.py", "line": 12, "comment": "The variable 'Age' has incorrect casing."}, 
          0.60)

# 2. Correct Keyword, Wrong Line
test_case("Keyword Only", 
          {"action_type": "comment", "file": "profiles.py", "line": 5, "comment": "The variable 'Age' is bad."}, 
          0.20)

# 3. Correct Line, No Keyword (Diagnosis missing)
test_case("Line Only", 
          {"action_type": "comment", "file": "profiles.py", "line": 12, "comment": "Something is wrong here on this line."}, 
          0.10)

# 4. Wrong everything (False Positive)
test_case("False Positive", 
          {"action_type": "comment", "file": "other.py", "line": 1, "comment": "This is totally fine code."}, 
          -0.30)

# 5. Correct Keyword + Line + Explanation Bonus
test_case("Perfect Match with Bonus", 
          {"action_type": "comment", "file": "profiles.py", "line": 12, "comment": "The variable 'Age' has a bug and fails to compile."}, 
          0.60)

print("\n--- All Precision Tests Passed! ---")
