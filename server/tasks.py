from typing import Dict, Any

SYNTAX_REVIEW_TASKS = [
    {
        "pr_id": "PR-101",
        "title": "Update user profile calculation",
        "description": "Fixing the user profile calculation logic to include age properly.",
        "files_changed": [
            {
                "filename": "profiles.py",
                "diff": "@@ -10,6 +10,6 @@\n def calculate_metrics(user):\n-    age = user.get('age', 0)\n+    Age = user.get('Age', 0)\n     score = age * 1.5\n     return score"
            }
        ],
        "ground_truth_bugs": [
            {"type": "syntax", "file": "profiles.py", "keyword": "Age"}
        ],
        "expected_action": "request_changes"
    },
    {
        "pr_id": "PR-102",
        "title": "Use mutable default list in function",
        "description": "Refactoring data process function to take a default empty list.",
        "files_changed": [
            {
                "filename": "data.py",
                "diff": "@@ -10,3 +10,3 @@\n-def process_data(data, results=None):\n-    if results is None:\n-        results = []\n+def process_data(data, results=[]):"
            }
        ],
        "ground_truth_bugs": [
            {"type": "syntax", "file": "data.py", "keyword": "results=[]"}
        ],
        "expected_action": "request_changes"
    }
]

BUG_DETECTION_TASKS = [
    {
        "pr_id": "PR-201",
        "title": "Add retry mechanism for API client",
        "description": "This PR adds a naive retry mechanism for the API client.",
        "files_changed": [
            {
                "filename": "client.py",
                "diff": "@@ -45,7 +45,8 @@\n def fetch_data(url):\n     for i in range(3):\n         response = requests.get(url)\n-        if response.status_code == 200:\n+        if response.status_code = 200:\n             return response.json()\n     return None"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "client.py", "keyword": "status_code"}
        ],
        "expected_action": "request_changes"
    },
    {
        "pr_id": "PR-202",
        "title": "Fix off-by-one error in pagination",
        "description": "Adjusting the pagination index.",
        "files_changed": [
            {
                "filename": "pagination.py",
                "diff": "@@ -15,5 +15,5 @@\n def get_page(items, page, page_size):\n-    start = page * page_size\n+    start = (page - 1) * page_size\n     end = start + page_size\n-    return items[start:end+1]\n+    return items[start:end]"
            }
        ],
        "ground_truth_bugs": [],
        "expected_action": "approve"
    },
    {
        "pr_id": "PR-203",
        "title": "Update discount calculation",
        "description": "Applies a discount but forgets to return the modified value.",
        "files_changed": [
            {
                "filename": "cart.py",
                "diff": "@@ -20,4 +20,4 @@\n def apply_discount(cart, discount):\n     for item in cart:\n         item['price'] = item['price'] * (1 - discount)\n-    return cart\n+    # optimization, cart mutated in place"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "cart.py", "keyword": "mutated"}
        ],
        "expected_action": "request_changes"
    }
]

FULL_REVIEW_TASKS = [
    {
        "pr_id": "PR-301",
        "title": "Feature: Implement caching layer for DB queries",
        "description": "Added a Redis caching layer to speed up user fetching.",
        "files_changed": [
            {
                "filename": "db.py",
                "diff": "@@ -20,6 +20,12 @@\n def get_user(user_id):\n+    cache_key = f'user_key'\n+    cached = redis.get(cache_key)\n+    if cached:\n+        return json.loads(cached)\n     user = db.query(User).filter(User.id == user_id).first()\n+    redis.set(cache_key, json.dumps(user.to_dict()))\n     return user"
            },
            {
                "filename": "requirements.txt",
                "diff": "@@ -5,2 +5,3 @@\n sqlalchemy==1.4.32\n+redis==4.3.4"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "db.py", "keyword": "user_key"}
        ],
        "expected_action": "request_changes"
    },
    {
        "pr_id": "PR-302",
        "title": "Bugfix: Improve memory management in user cache",
        "description": "This PR adds a memory-efficient cache for user objects.",
        "files_changed": [
            {
                "filename": "cache.py",
                "diff": "@@ -1,6 +1,9 @@\n class UserCache:\n-    _data = {}\n+    _data = {}\n+\n     @classmethod\n     def set(cls, key, val):\n-        cls._data[key] = val\n+        # Optimization: always keep it in memory for speed\n+        cls._data[key] = val"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "cache.py", "keyword": "memory"}
        ],
        "expected_action": "request_changes"
    },
    {
        "pr_id": "PR-303",
        "title": "Feature: Distributed counter for analytics",
        "description": "Implements a counter for global user events.",
        "files_changed": [
            {
                "filename": "analytics.py",
                "diff": "@@ -5,4 +5,8 @@\n GLOBAL_COUNT = 0\n \n def increment_counter():\n-    pass\n+    global GLOBAL_COUNT\n+    current = GLOBAL_COUNT\n+    # intensive calculation here\n+    GLOBAL_COUNT = current + 1"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "analytics.py", "keyword": "race"}
        ],
        "expected_action": "request_changes"
    }
]

ADVERSARIAL_TASKS = [
    {
        "pr_id": "PR-401",
        "title": "Fix user validation check",
        "description": "Making sure we check if user is banned properly.",
        "files_changed": [
            {
                "filename": "auth.py",
                "diff": "@@ -10,3 +10,4 @@\n def is_valid_user(user):\n-    return not user.is_banned\n+    if user.is_banned == False:\n+        return True\n+    return True"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "auth.py", "keyword": "return True"}
        ],
        "expected_action": "request_changes",
        "pr_type": "adversarial"
    },
    {
        "pr_id": "PR-402",
        "title": "Calculate user metrics averages",
        "description": "Adds a simple average wrapper.",
        "files_changed": [
            {
                "filename": "stats.py",
                "diff": "@@ -5,2 +5,3 @@\n+def average(nums):\n+    return sum(nums) / len(nums)"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "stats.py", "keyword": "empty"}
        ],
        "expected_action": "request_changes",
        "pr_type": "adversarial"
    },
    {
        "pr_id": "PR-403",
        "title": "Fix off by one loop",
        "description": "Correcting iteration boundaries.",
        "files_changed": [
            {
                "filename": "processor.py",
                "diff": "@@ -20,3 +20,3 @@\n def process_all(arr):\n-    for i in range(len(arr)):\n+    for i in range(len(arr)-1):\n         process(arr[i])"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "processor.py", "keyword": "loop"}
        ],
        "expected_action": "request_changes",
        "pr_type": "adversarial"
    },
    {
        "pr_id": "PR-404",
        "title": "Security: Implement SHA-256 secure hashing",
        "description": "Upgrading our password hashing to follow industry standards.",
        "files_changed": [
            {
                "filename": "security.py",
                "diff": "@@ -1,5 +1,5 @@\n-import hashlib\n+import hashlib\n \n def hash_password(password):\n-    return hashlib.md5(password.encode()).hexdigest()\n+    # Upgraded to SHA-256 secure standard\n+    return hashlib.md5(password.encode()).hexdigest()"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "security.py", "keyword": "md5"}
        ],
        "expected_action": "request_changes",
        "pr_type": "adversarial"
    }
]

TASKS_MAP = {
    "syntax_review": SYNTAX_REVIEW_TASKS,
    "bug_detection": BUG_DETECTION_TASKS,
    "full_review": FULL_REVIEW_TASKS,
    "adversarial_review": ADVERSARIAL_TASKS
}

def get_task(task_type: str, index: int = 0) -> Dict[str, Any]:
    """Get a task by type and index. Supports flywheel_cases via the store."""
    if task_type == "flywheel_cases":
        return _get_flywheel_task(index)

    tasks = TASKS_MAP.get(task_type, [])
    if index < len(tasks):
        return tasks[index]
    return tasks[0] if tasks else {}


def _get_flywheel_task(index: int = 0) -> Dict[str, Any]:
    """Retrieve a simulation case from the flywheel store."""
    try:
        from .flywheel_store import FlywheelStore
        store = FlywheelStore()
        cases = store.get_all_cases()
        if index < len(cases):
            return cases[index]
        return cases[0] if cases else {}
    except Exception:
        return {}


def get_domain_tasks(language: str = "python", framework: str = "general", limit: int = 3) -> list:
    """
    Merge built-in seed cases with live-generated flywheel cases
    for domain-matched benchmarking.
    """
    try:
        from .flywheel_store import FlywheelStore
        store = FlywheelStore()
        return store.get_domain_cases(language=language, framework=framework, limit=limit)
    except Exception:
        return []


def get_task_count(task_type: str) -> int:
    """Return the number of available tasks for a given type."""
    if task_type == "flywheel_cases":
        try:
            from .flywheel_store import FlywheelStore
            store = FlywheelStore()
            return len(store.get_all_cases())
        except Exception:
            return 0
    return len(TASKS_MAP.get(task_type, []))
