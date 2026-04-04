"""
GitHub PR Fetcher — Fetches real-world Pull Request data from public GitHub repos.
No dummy data, no assumptions. Pure GitHub REST API.
"""
import re
import httpx
from typing import Optional


def parse_pr_url(url: str) -> Optional[dict]:
    """
    Parses a GitHub PR URL into owner, repo, and PR number.
    Accepts: https://github.com/owner/repo/pull/123
    """
    pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.search(pattern, url)
    if not match:
        return None
    return {
        "owner": match.group(1),
        "repo": match.group(2),
        "pr_number": int(match.group(3))
    }


def fetch_pr_metadata(owner: str, repo: str, pr_number: int) -> dict:
    """Fetches PR title, description, author, merge status from GitHub API."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    resp = httpx.get(api_url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return {
        "title": data.get("title", ""),
        "description": data.get("body") or "No description provided.",
        "author": data["user"]["login"],
        "state": data.get("state", "unknown"),
        "mergeable": data.get("mergeable"),
        "mergeable_state": data.get("mergeable_state", "unknown"),
        "additions": data.get("additions", 0),
        "deletions": data.get("deletions", 0),
        "changed_files": data.get("changed_files", 0),
        "html_url": data.get("html_url", ""),
        "head_branch": data["head"]["ref"],
        "base_branch": data["base"]["ref"],
    }


def fetch_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    """Fetches the raw unified diff for the PR."""
    diff_url = f"https://github.com/{owner}/{repo}/pull/{pr_number}.diff"
    resp = httpx.get(diff_url, timeout=15, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def fetch_pr_files(owner: str, repo: str, pr_number: int) -> list:
    """Fetches the list of changed files with their patches."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    resp = httpx.get(api_url, timeout=15)
    resp.raise_for_status()
    files = resp.json()

    result = []
    for f in files:
        result.append({
            "filename": f["filename"],
            "status": f["status"],       # added, modified, removed
            "additions": f["additions"],
            "deletions": f["deletions"],
            "patch": f.get("patch", ""),  # unified diff for this file
        })
    return result


def fetch_full_pr(pr_url: str) -> dict:
    """
    One-shot function: Give it a GitHub PR URL, get back everything needed
    for the AI review. No dummy data — 100% real GitHub data.
    """
    parsed = parse_pr_url(pr_url)
    if not parsed:
        return {"error": f"Invalid GitHub PR URL: {pr_url}"}

    owner, repo, pr_number = parsed["owner"], parsed["repo"], parsed["pr_number"]

    try:
        metadata = fetch_pr_metadata(owner, repo, pr_number)
        files = fetch_pr_files(owner, repo, pr_number)
        diff = fetch_pr_diff(owner, repo, pr_number)
    except httpx.HTTPStatusError as e:
        return {"error": f"GitHub API error: {e.response.status_code} — {e.response.text}"}
    except Exception as e:
        return {"error": f"Failed to fetch PR: {str(e)}"}

    return {
        "pr_url": pr_url,
        "metadata": metadata,
        "files": files,
        "diff": diff,
    }
