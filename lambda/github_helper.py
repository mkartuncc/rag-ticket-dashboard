"""
GitHub API helper — creates PRs for tickets and labels/comments them after evaluation.
Uses only urllib (built-in), no extra packages needed.
"""
import json
import os
import urllib.request
import urllib.error

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO  = os.environ.get("GITHUB_REPO", "mkartuncc/rag-ticket-dashboard")
BASE_URL     = "https://api.github.com"


def _req(method: str, path: str, body: dict = None):
    url = f"{BASE_URL}/{path.lstrip('/')}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"GitHub API {method} {path} failed: {e.code} {e.read()}")
        raise


def _get_main_sha() -> str:
    data = _req("GET", f"repos/{GITHUB_REPO}/git/ref/heads/main")
    return data["object"]["sha"]


def create_ticket_pr(ticket_id: str, question: str, answer: str,
                     route: str, sender: str) -> int:
    """Create a branch + file + PR for a new ticket. Returns PR number."""
    branch = f"ticket/{ticket_id[:8]}"
    sha = _get_main_sha()

    # Create branch
    _req("POST", f"repos/{GITHUB_REPO}/git/refs", {
        "ref": f"refs/heads/{branch}",
        "sha": sha,
    })

    # Create ticket markdown file on the branch
    import base64
    content = (
        f"# Ticket `{ticket_id[:8]}`\n\n"
        f"**Question:** {question}\n\n"
        f"**Route:** `{route}`\n\n"
        f"**Sender:** {sender}\n\n"
        f"## Answer\n\n{answer}\n\n"
        f"---\n*Status: pending-eval*\n"
    )
    encoded = base64.b64encode(content.encode()).decode()
    _req("PUT", f"repos/{GITHUB_REPO}/contents/tickets/{ticket_id[:8]}.md", {
        "message": f"ticket: {question[:72]}",
        "content": encoded,
        "branch": branch,
    })

    # Open PR
    title = question if len(question) <= 72 else question[:69] + "..."
    pr = _req("POST", f"repos/{GITHUB_REPO}/pulls", {
        "title": title,
        "body": (
            f"**Sender:** {sender}\n"
            f"**Ticket ID:** `{ticket_id}`\n"
            f"**Route:** `{route}`\n\n"
            f"## Question\n{question}\n\n"
            f"## RAG Answer\n{answer}\n"
        ),
        "head": branch,
        "base": "main",
    })

    # Label as pending-eval
    _req("POST", f"repos/{GITHUB_REPO}/issues/{pr['number']}/labels", {
        "labels": ["pending-eval"],
    })

    return pr["number"]


def update_ticket_pr(pr_number: int, score: int, feedback: str, status: str):
    """Add eval comment and swap label on an existing PR."""
    # Post comment
    _req("POST", f"repos/{GITHUB_REPO}/issues/{pr_number}/comments", {
        "body": (
            f"## Evaluation Result\n\n"
            f"**Score:** {score}/10\n\n"
            f"**Status:** `{status}`\n\n"
            f"**Feedback:** {feedback}"
        ),
    })

    # Remove pending-eval label
    try:
        _req("DELETE", f"repos/{GITHUB_REPO}/issues/{pr_number}/labels/pending-eval")
    except Exception:
        pass

    # Add final label
    label = "approved" if status == "approved" else "needs-review"
    _req("POST", f"repos/{GITHUB_REPO}/issues/{pr_number}/labels", {
        "labels": [label],
    })
