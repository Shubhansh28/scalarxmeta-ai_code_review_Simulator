import requests
import json
import time

API_URL = "https://ashmit1812-scalarxmeta.hf.space"

def test_custom_pr():
    print(f"🌍 Connecting to {API_URL}...")
    
    # 1. Provide custom PR data (A deliberate bug!)
    custom_payload = {
        "title": "Fixing the auth loop",
        "description": "I added a quick bypass for local testing.",
        "files_changed": [
            {
                "filename": "auth.py",
                "diff": "--- auth.py\n+++ auth.py\n@@ -10,3 +10,3 @@\n def verify_admin(user):\n-    return user.is_admin and user.token_valid()\n+    return True  # TODO: revert this before pushing to production"
            }
        ],
        "expected_bugs": [
            {
                "file": "auth.py",
                "line": 12,
                "type": "Security",
                "description": "Hardcoded bypass for admin authorization."
            }
        ],
        "max_steps": 5
    }

    # Hit the /reset/custom endpoint
    print("\n🚀 [1] Sending Custom Challenge to /reset/custom...")
    response = requests.post(f"{API_URL}/reset/custom", json=custom_payload)
    if response.status_code != 200:
        print(f"❌ Error setting custom challenge: {response.text}")
        return

    data = response.json()
    session_id = data["session_id"]
    obs = data["observation"]
    print(f"✅ Success! Session ID: {session_id}")
    print(f"   Environment loaded PR: {obs['title']}")

    time.sleep(1)

    # 2. Try to spot the bug!
    print("\n🧐 [2] Agent takes an action: Commenting on the bug...")
    action_payload = {
        "session_id": session_id,
        "action": {
            "action_type": "comment",
            "comment": "There is a severe security vulnerability here. You hardcoded 'return True' which bypasses the admin check.",
            "file": "auth.py",
            "line": 12
        }
    }
    
    res_step = requests.post(f"{API_URL}/step", json=action_payload)
    step_data = res_step.json()
    
    print(f"   Reward received: {step_data['reward']}")
    print(f"   Feedback: {step_data['observation']['last_action_feedback']}")

    # 3. Finalize
    print("\n🛑 [3] Agent takes an action: Requesting Changes...")
    action2_payload = {
        "session_id": session_id,
        "action": {
            "action_type": "request_changes",
            "comment": "Cannot approve. Please remove the backdoor.",
            "file": None,
            "line": None
        }
    }
    res_step2 = requests.post(f"{API_URL}/step", json=action2_payload)
    step2_data = res_step2.json()
    
    print(f"   Final Decision Made. Environment Done: {step2_data['done']}")
    print(f"🏆 Final AI Grader Score: {step2_data['info'].get('score', 0):.2f}/1.0")

if __name__ == "__main__":
    test_custom_pr()
