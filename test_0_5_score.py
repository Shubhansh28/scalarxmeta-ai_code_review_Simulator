import requests
import time

API_URL = "http://localhost:7860"

def test_0_5_score():
    print(f"🌍 Connecting to {API_URL}...")
    
    # 1. Reset Environment (Use default 'syntax_review' index 0)
    print("\n🚀 [1] Initializing 'syntax_review' task (Index 1)...")
    res_reset = requests.post(f"{API_URL}/reset", json={"task_type": "syntax_review", "task_index": 1})
    if res_reset.status_code != 200:
        print("❌ Error resetting environment.")
        return
    data = res_reset.json()
    session_id = data["session_id"]
    obs = data["observation"]
    print(f"✅ Success! Session ID: {session_id}")
    print(f"   Task Loaded: {obs['title']}")

    time.sleep(1)

    # 2. Step 1: Find the bug WITH explanation bonus (+0.4 base + 0.2 exp - 0.05 step = +0.55)
    print("\n🧐 [2] Action 1: High Quality Comment (Finds bug + explanation)")
    res1 = requests.post(f"{API_URL}/step", json={
        "session_id": session_id,
        "action": {
            "action_type": "comment",
            "comment": "The mutable default argument is a serious logic bug.",
            "file": "data.py",
            "line": 10
        }
    }).json()
    print(f"   Reward: {res1['reward']:.2f} | Current Score: {res1['info'].get('score', 0):.2f}")

    time.sleep(1)

    # 3. Step 2: Short comment penalty (-0.2 base - 0.05 step = -0.25)
    print("\n🧐 [3] Action 2: Short/Spam Comment (< 5 words)")
    res2 = requests.post(f"{API_URL}/step", json={
        "session_id": session_id,
        "action": {
            "action_type": "comment",
            "comment": "short comment",
            "file": "main.py",
            "line": 1
        }
    }).json()
    print(f"   Reward: {res2['reward']:.2f} | Current Score: {res2['info'].get('score', 0):.2f}")

    time.sleep(1)

    # 4. Step 3: Another Short comment penalty (-0.2 base - 0.05 step = -0.25)
    print("\n🧐 [4] Action 3: Another Short/Spam Comment (< 5 words)")
    res3 = requests.post(f"{API_URL}/step", json={
        "session_id": session_id,
        "action": {
            "action_type": "comment",
            "comment": "short comment",
            "file": "main.py",
            "line": 1
        }
    }).json()
    print(f"   Reward: {res3['reward']:.2f} | Current Score: {res3['info'].get('score', 0):.2f}")

    time.sleep(1)

    # 5. Step 4: Request Changes (+0.5 correct decision - 0.05 step = +0.45)
    print("\n🛑 [5] Action 4: Request Changes (Final Decision)")
    res4 = requests.post(f"{API_URL}/step", json={
        "session_id": session_id,
        "action": {
            "action_type": "request_changes",
            "comment": "Please fix the bugs.",
        }
    }).json()
    print(f"   Reward: {res4['reward']:.2f} | Final Done State: {res4['done']}")
    
    print("\n" + "="*40)
    print(f"🏆 Final AI Grader Score: {res4['info'].get('score', 0):.2f} / 1.00")
    print("="*40)

if __name__ == "__main__":
    test_0_5_score()
