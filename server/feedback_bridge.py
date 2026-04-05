"""
Feedback Bridge — Connects Live PR Review output to the Simulator input.
Handles signal capture, multi-signal validation, PR-to-simulation conversion,
and privacy stripping.
"""
import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Language Detection ───────────────────────────────────────

EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".css": "css",
    ".html": "html",
}


def detect_language(files: List[Dict]) -> str:
    """Detect the dominant language from file extensions."""
    lang_counts: Dict[str, int] = {}
    for f in files:
        filename = f.get("filename", "")
        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
        lang = EXTENSION_MAP.get(ext, "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    if not lang_counts:
        return "python"
    return max(lang_counts, key=lang_counts.get)


def detect_framework(files: List[Dict], diff_text: str = "") -> str:
    """Detect framework from filenames and diff content."""
    combined = " ".join(f.get("filename", "") for f in files) + " " + diff_text

    framework_signals = {
        "react": ["jsx", "tsx", "useEffect", "useState", "React"],
        "nextjs": ["next.config", "getServerSideProps", "getStaticProps"],
        "django": ["django", "models.py", "views.py", "urls.py"],
        "flask": ["flask", "app.route", "@app"],
        "fastapi": ["fastapi", "FastAPI", "Depends"],
        "express": ["express", "app.get(", "app.post("],
    }

    for framework, signals in framework_signals.items():
        if any(s.lower() in combined.lower() for s in signals):
            return framework

    return "general"


# ── Signal Validation ────────────────────────────────────────

def should_convert(signals: List[Dict]) -> Tuple[bool, List[int]]:
    """
    Multi-signal validation: a bug must be both AI-flagged AND 
    developer-confirmed before it becomes a simulation case.
    
    Returns (should_convert, list_of_confirmed_bug_indices).
    """
    confirmed_indices = []

    for signal in signals:
        if signal.get("signal_type") == "confirm_bug" and signal.get("bug_index") is not None:
            confirmed_indices.append(signal["bug_index"])

    # Need at least one confirmed bug
    return len(confirmed_indices) > 0, confirmed_indices


# ── Privacy Stripping ────────────────────────────────────────

def strip_business_logic(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Anonymize variable names and business-specific identifiers,
    retaining only the structural bug pattern.
    """
    stripped = dict(case)

    # Patterns to anonymize: CamelCase identifiers, snake_case vars
    var_counter = [0]
    seen_vars: Dict[str, str] = {}

    def _replace_var(match):
        original = match.group(0)
        # Skip language keywords and common stdlib names
        skip = {
            "def", "return", "if", "else", "for", "while", "class", "import",
            "from", "None", "True", "False", "self", "cls", "print", "len",
            "range", "list", "dict", "str", "int", "float", "bool", "async",
            "await", "function", "const", "let", "var", "null", "undefined",
            "func", "package", "type", "struct", "err", "error",
        }
        if original in skip or len(original) <= 2:
            return original
        if original not in seen_vars:
            var_counter[0] += 1
            seen_vars[original] = f"var_{var_counter[0]}"
        return seen_vars[original]

    for file_change in stripped.get("files_changed", []):
        diff = file_change.get("diff", "")
        # Replace snake_case and camelCase identifiers (3+ chars)
        diff = re.sub(r'\b[a-z][a-z_]{2,}\b', _replace_var, diff)
        file_change["diff"] = diff

    # Anonymize title and description
    stripped["title"] = re.sub(r'\b[A-Z][a-zA-Z]+\b', "Component", stripped.get("title", ""))
    stripped["description"] = "Converted from live PR review."

    return stripped


# ── PR → Simulation Conversion ───────────────────────────────

def convert_to_simulation_case(
    pr_data: Dict[str, Any],
    ai_result: Dict[str, Any],
    confirmed_bug_indices: List[int],
    strip_private: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Convert a live PR with confirmed bugs into a simulation case
    that can be used by the OpenEnv environment.
    """
    if not confirmed_bug_indices:
        return None

    metadata = pr_data.get("metadata", {})
    files = pr_data.get("files", [])
    ai_comments = ai_result.get("comments", [])

    # Build ground_truth_bugs from confirmed AI findings
    ground_truth_bugs = []
    for idx in confirmed_bug_indices:
        if idx < len(ai_comments):
            comment = ai_comments[idx]
            # Extract a keyword from the comment for the grader
            keyword = _extract_keyword(comment.get("comment", ""))
            ground_truth_bugs.append({
                "type": comment.get("severity", "logic"),
                "file": comment.get("file", "unknown"),
                "keyword": keyword,
            })

    if not ground_truth_bugs:
        return None

    # Build files_changed in environment format
    files_changed = []
    for f in files:
        files_changed.append({
            "filename": f.get("filename", ""),
            "diff": f.get("patch", ""),
        })

    # Detect language and framework
    all_diffs = " ".join(f.get("patch", "") for f in files)
    language = detect_language(files)
    framework = detect_framework(files, all_diffs)

    case = {
        "pr_id": f"LIVE-{uuid.uuid4().hex[:6].upper()}",
        "title": metadata.get("title", "Live PR Review"),
        "description": metadata.get("description", "Converted from live review."),
        "files_changed": files_changed,
        "ground_truth_bugs": ground_truth_bugs,
        "expected_action": "request_changes",
        "source": "live_confirmed",
        "language": language,
        "framework": framework,
    }

    if strip_private:
        case = strip_business_logic(case)

    return case


def _extract_keyword(comment_text: str) -> str:
    """
    Extract the most diagnostic keyword from an AI review comment.
    Used as the grader's matching keyword for the new simulation case.
    """
    # Prioritize known diagnostic terms
    diagnostic_terms = [
        "race condition", "sql injection", "xss", "buffer overflow",
        "null pointer", "division by zero", "memory leak", "deadlock",
        "off-by-one", "use after free", "uninitialized", "overflow",
        "injection", "bypass", "hardcoded", "credentials", "token",
        "md5", "sha1", "plaintext", "return true", "return false",
        "mutable default", "empty list", "missing await", "missing return",
        "type coercion", "loose equality", "unchecked error",
    ]

    comment_lower = comment_text.lower()
    for term in diagnostic_terms:
        if term in comment_lower:
            return term

    # Fallback: longest meaningful word from the comment
    words = [w for w in comment_text.split() if len(w) > 4 and w.isalpha()]
    if words:
        return max(words, key=len).lower()

    return "issue"


# ── Capture Signal ───────────────────────────────────────────

def capture_developer_signal(
    store,
    session_id: str,
    signal_type: str,
    bug_index: Optional[int] = None,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record a developer's feedback on a live review finding.
    If enough confirmations accumulate, triggers conversion.
    """
    signal = {
        "signal_type": signal_type,
        "bug_index": bug_index,
        "comment": comment,
    }

    store.add_signal_to_session(session_id, signal)
    session = store.get_review_session(session_id)

    result = {"recorded": True, "signal": signal, "converted": False}

    if session:
        ai_result = session.get("ai_result", {})
        ai_comments = ai_result.get("comments", [])

        # Update pattern stats based on signal type
        if signal_type == "confirm_bug" and bug_index is not None and bug_index < len(ai_comments):
            keyword = _extract_keyword(ai_comments[bug_index].get("comment", ""))
            store.record_confirmation(keyword)

        elif signal_type == "dismiss" and bug_index is not None and bug_index < len(ai_comments):
            keyword = _extract_keyword(ai_comments[bug_index].get("comment", ""))
            store.record_dismissal(keyword)

        # Check if we should convert to simulation case
        can_convert, confirmed_indices = should_convert(session.get("signals", []))
        if can_convert:
            case = convert_to_simulation_case(
                session["pr_data"],
                session["ai_result"],
                confirmed_indices,
            )
            if case:
                case_id = store.add_simulation_case(case)
                result["converted"] = True
                result["case_id"] = case_id

    return result
