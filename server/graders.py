import logging
from typing import Any, Tuple, Set, Dict

logger = logging.getLogger(__name__)

# --- Grading Constants ---
MIN_COMMENT_WORDS = 5
EXPLANATION_KEYWORDS = ["edge case", "bug", "fails", "incorrect", "crash", "security", "logic", "leak"]

def evaluate_step(task_data: Dict[str, Any], action: Any, identified_bugs: Set[int]) -> Tuple[float, Set[int]]:
    """
    Evaluates a single action and returns (reward, newly_identified_bug_indices).
    
    Rules:
    - Correct bug detection (Keyword + Line): +0.6 (Full Precision)
    - Correct keyword but wrong/missing line: +0.2 (Partial progress)
    - Correct line but wrong/missing keyword: +0.1 (Right location, wrong diagnosis)
    - False positive: -0.3
    - Short/spam comment: -0.2
    - Wrong file: -0.3 (Strict file validation)
    - Unjustified approval/rejection: -0.5
    """
    reward = 0.0
    new_bugs = set()
    
    ground_truth_bugs = task_data.get("ground_truth_bugs", [])
    expected_action = task_data.get("expected_action", "approve")
    
    # Handle both dict and object-style actions
    action_type = getattr(action, "action_type", action.get("action_type") if isinstance(action, dict) else None)
    comment = getattr(action, "comment", action.get("comment", "") if isinstance(action, dict) else "")
    file_name = getattr(action, "file", action.get("file") if isinstance(action, dict) else None)
    line = getattr(action, "line", action.get("line") if isinstance(action, dict) else None)
    
    if action_type == "comment":
        # 1. Quality Check: Minimum words
        words = comment.split()
        if len(words) < MIN_COMMENT_WORDS:
            return -0.2, set() # Short/spam penalty
        
        # 2. Match Analysis
        comment_lower = comment.lower()
        hit_idx = -1
        found_partial_match = False
        
        for i, bug in enumerate(ground_truth_bugs):
            if i in identified_bugs:
                continue
            
            # Robust matching: Case-insensitive keyword check
            keyword = bug["keyword"].lower()
            target_line = bug.get("line")
            
            keyword_match = keyword in comment_lower
            line_match = (file_name == bug.get("file")) and (line == target_line) if (line is not None and target_line is not None) else False
            
            if keyword_match and line_match:
                hit_idx = i
                reward += 0.6 # Full credit for precision
                break
            elif keyword_match:
                # Found the keyword but missed the line or file
                if file_name and file_name != bug.get("file"):
                    reward -= 0.3
                else:
                    reward += 0.2 # Partial credit for correct bug identification
                hit_idx = i
                break
            elif line_match:
                # Found the line but missed the keyword (diagnosis)
                reward += 0.1
                found_partial_match = True
                # Note: We don't mark the bug as "identified" yet if the keyword is missing
                continue
        
        if hit_idx != -1:
            new_bugs.add(hit_idx)
            # 4. Explanation Bonus
            if any(kw in comment_lower for kw in EXPLANATION_KEYWORDS):
                reward += 0.1
        elif found_partial_match:
            # Already added 0.1, don't penalize as false positive
            pass
        else:
            # False Positive
            reward -= 0.3
            
    elif action_type in ["approve", "request_changes"]:
        # 5. Justification Check
        if action_type == expected_action:
            # If requesting changes, MUST have found at least one bug in this session
            if action_type == "request_changes" and not identified_bugs and not new_bugs:
                reward -= 0.5 # Unjustified request_changes
            else:
                reward += 0.5 # Correct and justified
        else:
            # Catastrophic error (approving bad PR or rejecting perfect PR)
            if expected_action == "request_changes" and action_type == "approve":
                reward -= 0.6 # Extremely bad: approving buggy code
            else:
                reward -= 0.4 # General wrong decision
                
    return reward, new_bugs

def finalize_episode(task_data: Dict[str, Any], identified_bugs: Set[int]) -> float:
    """
    Applies end-of-episode penalties for missed bugs.
    """
    ground_truth_bugs = task_data.get("ground_truth_bugs", [])
    missed_count = len(ground_truth_bugs) - len(identified_bugs)
    
    if missed_count > 0:
        # Scale penalty: -0.4 for first bug, -0.5 for subsequent
        return -(0.4 + (missed_count - 1) * 0.5)
    
    return 0.0
