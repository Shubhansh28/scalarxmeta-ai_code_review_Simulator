"""
Flywheel Store — Persistent, self-updating simulation library.
Ships with a seed library of general-purpose cases. Grows as live PR bugs
are confirmed by developers.
"""
import json
import os
import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "flywheel_store.json")

# ── Seed Library ─────────────────────────────────────────────
# Ships with the project to solve the cold-start problem.

SEED_CASES = [
    # Python — Mutable default argument
    {
        "case_id": "seed-py-001",
        "pr_id": "SEED-PY-001",
        "title": "Refactor data pipeline defaults",
        "description": "Updated function to use a default list parameter.",
        "files_changed": [
            {
                "filename": "pipeline.py",
                "diff": "@@ -5,3 +5,2 @@\n-def run_pipeline(data, results=None):\n-    if results is None:\n-        results = []\n+def run_pipeline(data, results=[]):"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "pipeline.py", "keyword": "results=[]"}
        ],
        "expected_action": "request_changes",
        "source": "seed",
        "language": "python",
        "framework": "general",
        "created_at": "2026-01-01T00:00:00"
    },
    # Python — Division by zero
    {
        "case_id": "seed-py-002",
        "pr_id": "SEED-PY-002",
        "title": "Add average calculation utility",
        "description": "Simple average helper for metrics dashboard.",
        "files_changed": [
            {
                "filename": "metrics.py",
                "diff": "@@ -1,2 +1,3 @@\n+def compute_average(values):\n+    return sum(values) / len(values)"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "metrics.py", "keyword": "empty"}
        ],
        "expected_action": "request_changes",
        "source": "seed",
        "language": "python",
        "framework": "general",
        "created_at": "2026-01-01T00:00:00"
    },
    # JavaScript — Triple equals
    {
        "case_id": "seed-js-001",
        "pr_id": "SEED-JS-001",
        "title": "Fix user role check",
        "description": "Updated role comparison in auth middleware.",
        "files_changed": [
            {
                "filename": "auth.js",
                "diff": "@@ -10,3 +10,3 @@\n function isAdmin(user) {\n-    return user.role === 'admin';\n+    return user.role == 'admin';\n }"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "auth.js", "keyword": "=="}
        ],
        "expected_action": "request_changes",
        "source": "seed",
        "language": "javascript",
        "framework": "general",
        "created_at": "2026-01-01T00:00:00"
    },
    # JavaScript — Missing await
    {
        "case_id": "seed-js-002",
        "pr_id": "SEED-JS-002",
        "title": "Fetch user data from API",
        "description": "Added async user fetch for the profile page.",
        "files_changed": [
            {
                "filename": "profile.js",
                "diff": "@@ -5,3 +5,5 @@\n async function loadProfile(userId) {\n-    const user = await fetch(`/api/users/${userId}`);\n+    const user = fetch(`/api/users/${userId}`);\n+    const data = user.json();\n+    return data;\n }"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "profile.js", "keyword": "await"}
        ],
        "expected_action": "request_changes",
        "source": "seed",
        "language": "javascript",
        "framework": "general",
        "created_at": "2026-01-01T00:00:00"
    },
    # Go — Unchecked error
    {
        "case_id": "seed-go-001",
        "pr_id": "SEED-GO-001",
        "title": "Read config file on startup",
        "description": "Load configuration from JSON file at boot.",
        "files_changed": [
            {
                "filename": "config.go",
                "diff": "@@ -10,4 +10,4 @@\n func LoadConfig(path string) Config {\n-    data, err := os.ReadFile(path)\n-    if err != nil { log.Fatal(err) }\n+    data, _ := os.ReadFile(path)\n     var cfg Config\n     json.Unmarshal(data, &cfg)"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "config.go", "keyword": "error"}
        ],
        "expected_action": "request_changes",
        "source": "seed",
        "language": "go",
        "framework": "general",
        "created_at": "2026-01-01T00:00:00"
    },
    # Go — Race condition on shared map
    {
        "case_id": "seed-go-002",
        "pr_id": "SEED-GO-002",
        "title": "Add concurrent cache for sessions",
        "description": "Shared map for caching active sessions.",
        "files_changed": [
            {
                "filename": "cache.go",
                "diff": "@@ -5,6 +5,8 @@\n var sessions = map[string]Session{}\n \n func SetSession(id string, s Session) {\n     sessions[id] = s\n }\n func GetSession(id string) Session {\n     return sessions[id]\n }"
            }
        ],
        "ground_truth_bugs": [
            {"type": "logic", "file": "cache.go", "keyword": "race"}
        ],
        "expected_action": "request_changes",
        "source": "seed",
        "language": "go",
        "framework": "general",
        "created_at": "2026-01-01T00:00:00"
    },
]


class FlywheelStore:
    """
    JSON-backed persistent store for the self-learning flywheel.
    Holds simulation cases and pattern-level statistics.
    """

    def __init__(self, path: str = STORE_PATH):
        self.path = path
        self.cases: List[Dict[str, Any]] = []
        self.pattern_stats: Dict[str, Dict[str, Any]] = {}
        self.review_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> live review data
        self.load()

    def load(self):
        """Load store from disk. If missing, seed with defaults."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.cases = data.get("cases", [])
                self.pattern_stats = data.get("pattern_stats", {})
                logger.info(f"Flywheel store loaded: {len(self.cases)} cases")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Corrupt store, reinitializing: {e}")
                self._initialize_with_seeds()
        else:
            self._initialize_with_seeds()

    def _initialize_with_seeds(self):
        """Populate with seed library on first run."""
        self.cases = [dict(c) for c in SEED_CASES]
        self.pattern_stats = {}
        self.save()
        logger.info(f"Flywheel store initialized with {len(SEED_CASES)} seed cases")

    def save(self):
        """Persist current state to disk."""
        data = {
            "cases": self.cases,
            "pattern_stats": self.pattern_stats,
            "last_updated": datetime.utcnow().isoformat()
        }
        try:
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save flywheel store: {e}")

    # ── Case Management ──────────────────────────────────────

    def add_simulation_case(self, case_data: Dict[str, Any]) -> str:
        """
        Add a confirmed live PR as a new simulation case.
        Returns the new case_id.
        """
        case_id = f"live-{uuid.uuid4().hex[:8]}"
        case_data["case_id"] = case_id
        case_data["source"] = "live_confirmed"
        case_data["created_at"] = datetime.utcnow().isoformat()
        self.cases.append(case_data)
        self.save()
        logger.info(f"Added flywheel case: {case_id}")
        return case_id

    def get_domain_cases(
        self,
        language: str = "python",
        framework: str = "general",
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Return simulation cases matching a PR's domain."""
        matches = [
            c for c in self.cases
            if c.get("language", "").lower() == language.lower()
        ]
        # Prefer live-confirmed over seed
        matches.sort(key=lambda c: (c.get("source") == "seed", c.get("created_at", "")))
        return matches[:limit]

    def get_all_cases(self) -> List[Dict[str, Any]]:
        """Return the full simulation library."""
        return list(self.cases)

    # ── Pattern Statistics ───────────────────────────────────

    def record_flag(self, keyword: str):
        """Record that the AI flagged a pattern."""
        stats = self.pattern_stats.setdefault(keyword, {
            "keyword": keyword,
            "times_flagged": 0,
            "times_confirmed": 0,
            "times_dismissed": 0,
            "decay_weight": 1.0,
        })
        stats["times_flagged"] += 1
        self._recompute_accuracy(keyword)
        self.save()

    def record_confirmation(self, keyword: str):
        """Record that a developer confirmed an AI-flagged bug."""
        stats = self.pattern_stats.setdefault(keyword, {
            "keyword": keyword,
            "times_flagged": 0,
            "times_confirmed": 0,
            "times_dismissed": 0,
            "decay_weight": 1.0,
        })
        stats["times_confirmed"] += 1
        self._recompute_accuracy(keyword)
        self.save()

    def record_dismissal(self, keyword: str):
        """Record that a developer dismissed an AI flag as a false positive."""
        stats = self.pattern_stats.setdefault(keyword, {
            "keyword": keyword,
            "times_flagged": 0,
            "times_confirmed": 0,
            "times_dismissed": 0,
            "decay_weight": 1.0,
        })
        stats["times_dismissed"] += 1
        # Decay mechanism: each dismissal reduces weight by 10%
        stats["decay_weight"] = max(0.1, stats["decay_weight"] * 0.9)
        self._recompute_accuracy(keyword)
        self.save()

    def _recompute_accuracy(self, keyword: str):
        stats = self.pattern_stats.get(keyword)
        if stats and stats["times_flagged"] > 0:
            stats["accuracy"] = round(
                stats["times_confirmed"] / stats["times_flagged"] * 100, 1
            )

    def get_pattern_stats(self, keyword: str) -> Optional[Dict[str, Any]]:
        """Get historical accuracy for a particular bug pattern."""
        return self.pattern_stats.get(keyword)

    def get_all_pattern_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return all pattern statistics."""
        return dict(self.pattern_stats)

    # ── Review Session Tracking ──────────────────────────────

    def register_review_session(self, session_id: str, pr_data: dict, ai_result: dict):
        """Track a live review session for later signal capture."""
        self.review_sessions[session_id] = {
            "pr_data": pr_data,
            "ai_result": ai_result,
            "signals": [],
            "created_at": datetime.utcnow().isoformat()
        }

    def get_review_session(self, session_id: str) -> Optional[Dict]:
        return self.review_sessions.get(session_id)

    def add_signal_to_session(self, session_id: str, signal: dict):
        session = self.review_sessions.get(session_id)
        if session:
            session["signals"].append(signal)

    # ── Library Stats ────────────────────────────────────────

    def get_library_stats(self) -> Dict[str, Any]:
        """Summary statistics for the flywheel dashboard."""
        total = len(self.cases)
        seed_count = sum(1 for c in self.cases if c.get("source") == "seed")
        live_count = total - seed_count

        by_language: Dict[str, int] = {}
        for c in self.cases:
            lang = c.get("language", "unknown")
            by_language[lang] = by_language.get(lang, 0) + 1

        recent = sorted(
            self.cases,
            key=lambda c: c.get("created_at", ""),
            reverse=True
        )[:10]

        return {
            "total_cases": total,
            "seed_cases": seed_count,
            "live_cases": live_count,
            "by_language": by_language,
            "recent_cases": [
                {"case_id": c["case_id"], "title": c["title"], "source": c["source"]}
                for c in recent
            ],
            "total_patterns_tracked": len(self.pattern_stats),
        }

    # ── Export / Import (persistence across redeploys) ────────

    def export_data(self) -> Dict[str, Any]:
        """Export the full store for backup."""
        return {
            "cases": self.cases,
            "pattern_stats": self.pattern_stats,
            "exported_at": datetime.utcnow().isoformat()
        }

    def import_data(self, data: Dict[str, Any]):
        """Import a previously exported store."""
        self.cases = data.get("cases", [])
        self.pattern_stats = data.get("pattern_stats", {})
        self.save()
        logger.info(f"Imported flywheel store: {len(self.cases)} cases")
