from redis_memory import (
    check_redis_connection,
    save_memory,
    get_memory
)


if __name__ == "__main__":

    if check_redis_connection():

        save_memory(
            "user_1",
            "user",
            "Create a GitHub repository named DevPulseTest"
        )

        save_memory(
            "user_1",
            "assistant",
            "Repository creation request received"
        )

        messages = get_memory("user_1")

        print("\nStored Messages:")

        for message in messages:
            print(message)






"""
=============================================================
DevPulse — Agent Core
Model + Tools + Memory + Human-in-the-Loop (HITL)

Flow:

READ TOOL:
User
   ↓
LLM (Reason)
   ↓
Select Read Tool
   ↓
MCP Tool
   ↓
GitHub API
   ↓
LLM Response

WRITE TOOL:
User
   ↓
LLM (Reason)
   ↓
Select Write Tool
   ↓
Human Approval
   ↓
MCP Tool
   ↓
GitHub API
   ↓
LLM Response
=============================================================
"""

import json
import asyncio

from model_client import get_client_and_model, check_connection
from mcp_bridge import mcp_session, discover_openai_tools, call_mcp_tool


# =============================================================
# TOOLS THAT REQUIRE HUMAN APPROVAL
# =============================================================

APPROVAL_REQUIRED_TOOLS = {
    "create_repository",
    "delete_repository",
    "edit_repository",
    "create_file",
    "update_file",
    "delete_file",
    "create_branch",
    "delete_branch",
    "create_issue",
    "close_issue",
    "create_pull_request",
    "merge_pull_request",
}


# =============================================================
# SYSTEM PROMPT
# =============================================================

SYSTEM_PROMPT = (
    
    "You are DevPulse, an AI-powered GitHub automation assistant. "

    "You have access to real, live GitHub tools. "
    "Always use the appropriate tool instead of guessing GitHub information. "

    "For read-only requests, use the appropriate read-only tool. "

    "For requests that modify GitHub data, use the appropriate action tool. "

    "Remember repository names and other relevant information mentioned "
    "earlier in the conversation. "

    "IMPORTANT: When the user requests multiple actions, you MUST "
    "complete ALL requested actions before giving the final answer. "

    "Execute the required tools sequentially. "

    "After each tool result, check whether any part of the user's original "
    "request remains incomplete. If something remains, call the appropriate "
    "tool instead of returning a final answer. "

    "For example, if the user asks to create a repository and then create "
    "files inside it, first create the repository, then use the returned "
    "repository information to create each requested file. "

    "Do not stop after completing only the first action."
)



# =============================================================
# DEVPULSE AGENT
# =============================================================

class DevPulseAgent:

    def __init__(self):
        self.client, self.model, self.backend = get_client_and_model()
        self.history = []

    # =========================================================
    # ASK
    # =========================================================

    async def ask(self, user_input: str, max_loops: int = 6) -> str:

        # -----------------------------------------------------
        # Store user message
        # -----------------------------------------------------

        self.history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        # -----------------------------------------------------
        # Connect to MCP server
        # -----------------------------------------------------

        async with mcp_session() as session:

            # -------------------------------------------------
            # Discover MCP tools dynamically
            # -------------------------------------------------

            tools = await discover_openai_tools(session)

            # -------------------------------------------------
            # Agent reasoning loop
            # -------------------------------------------------

            for _ in range(max_loops):

                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                ] + self.history

                # -------------------------------------------------
                # Ask LLM
                # -------------------------------------------------

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # -------------------------------------------------
                # No tool required → final response
                # -------------------------------------------------

                if finish_reason != "tool_calls" or not msg.tool_calls:

                    final_content = msg.content or ""

                    self.history.append(
                        {
                            "role": "assistant",
                            "content": final_content
                        }
                    )

                    return final_content

                # -------------------------------------------------
                # Store assistant tool request
                # -------------------------------------------------

                self.history.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            tc.model_dump()
                            for tc in msg.tool_calls
                        ]
                    }
                )

                # -------------------------------------------------
                # Execute requested tools
                # -------------------------------------------------

                for tool_call in msg.tool_calls:

                    tool_name = tool_call.function.name

                    try:
                        args = json.loads(
                            tool_call.function.arguments
                        )
                    except json.JSONDecodeError:

                        result_text = (
                            f"Invalid tool arguments generated "
                            f"for {tool_name}."
                        )

                        self.history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_text
                            }
                        )

                        continue

                    print("\n")
                    print("=" * 60)
                    print(" TOOL REQUEST ")
                    print("=" * 60)
                    print(f"Tool Selected : {tool_name}")
                    print(f"Arguments     : {args}")
                    print("=" * 60)

                    # =================================================
                    # CHECK APPROVAL
                    # =================================================

                    if tool_name in APPROVAL_REQUIRED_TOOLS:

                        # -------------------------------------------------
                        # WRITE / ACTION TOOL
                        # -------------------------------------------------

                        print("\n")
                        print("=" * 60)
                        print(" HUMAN-IN-THE-LOOP APPROVAL ")
                        print("=" * 60)

                        print(f"Tool Selected : {tool_name}")
                        print(f"Arguments     : {args}")

                        print("=" * 60)

                        approval = input(
                            "Approve tool execution? (y/n): "
                        ).strip().lower()

                        # -------------------------------------------------
                        # User rejected tool
                        # -------------------------------------------------

                        if approval != "y":

                            print(
                                "\n❌ Tool execution rejected "
                                "by human.\n"
                            )

                            result_text = (
                                "Tool execution denied by human reviewer."
                            )

                            self.history.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": result_text
                                }
                            )

                            continue

                        print("\n✅ Human approved.")

                    else:

                        # -------------------------------------------------
                        # READ-ONLY TOOL
                        # -------------------------------------------------

                        print(
                            "\n📖 Read-only tool detected."
                        )

                        print(
                            f"Executing '{tool_name}' "
                            "without approval..."
                        )

                    # =================================================
                    # EXECUTE MCP TOOL
                    # =================================================

                    print("\nExecuting MCP Tool...\n")

                    try:

                        result_text = await call_mcp_tool(
                            session,
                            tool_name,
                            args
                        )

                    except Exception as e:

                        result_text = (
                            f"Error executing tool "
                            f"{tool_name}: {str(e)}"
                        )

                    # =================================================
                    # DISPLAY TOOL RESULT
                    # =================================================

                    print("=" * 60)
                    print(" TOOL RESULT ")
                    print("=" * 60)

                    print(result_text)

                    print("=" * 60)

                    # =================================================
                    # SEND RESULT BACK TO LLM
                    # =================================================

                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text
                        }
                    )

            # -----------------------------------------------------
            # Maximum reasoning loops reached
            # -----------------------------------------------------

            return (
                "I wasn't able to resolve this request within "
                "the allotted reasoning steps."
            )


# =============================================================
# MAIN
# =============================================================

async def main():

    agent = DevPulseAgent()

    # ---------------------------------------------------------
    # Check model connection
    # ---------------------------------------------------------

    if not check_connection(
        agent.client,
        agent.model,
        agent.backend
    ):
        return

    print("\n")
    print("=" * 70)
    print(" DevPulse — GitHub Automation Agent ")
    print("=" * 70)

    print(f"Model : {agent.model}")
    

    print("Type 'quit' to exit.")

    print("=" * 70)

    # ---------------------------------------------------------
    # Chat loop
    # ---------------------------------------------------------

    while True:

        user_input = input("\nYou : ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        answer = await agent.ask(user_input)

        print("\n")
        print("=" * 70)
        print(" DevPulse Response ")
        print("=" * 70)

        print(answer)

        print("=" * 70)


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":
    asyncio.run(main())







from dotenv import load_dotenv
 
import os
import httpx
import base64
from mcp.server.fastmcp import FastMCP


load_dotenv()
# -- SETUP ---------------------------------------------------------------
mcp = FastMCP("DevPulse GitHub Intelligence Server")
 
GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # optional, raises rate limit 60 -> 5000/hr

print("GitHub Token:", GITHUB_TOKEN)
# GitHub requires a User-Agent header on every request, or it rejects the call.
_HEADERS = {"User-Agent": "DevPulse-MCP-Server"}

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "DevPulse"
}

if GITHUB_TOKEN:
    _HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

print(_HEADERS)



def _get_authenticated_user() -> str:
    """
    Get the username of the authenticated GitHub user.
    """

    url = f"{GITHUB_API}/user"

    try:
        response = httpx.get(
            url,
            headers=_HEADERS,
            timeout=10,
            follow_redirects=True
        )
    except httpx.RequestError as e:
        raise RuntimeError(
            f"Could not connect to GitHub: {e}"
        )

    if response.status_code != 200:
        try:
            data = response.json()
            message = data.get("message", response.text)
        except Exception:
            message = response.text

        raise RuntimeError(
            f"Could not determine GitHub user: {message}"
        )

    data = response.json()

    username = data.get("login")

    if not username:
        raise RuntimeError(
            "GitHub did not return the authenticated username."
        )

    return username



 
def _get(path: str, params: dict | None = None) -> tuple[int, dict]:
    """Single shared real HTTP GET against the GitHub REST API."""
    try:
        r = httpx.get(
    f"{GITHUB_API}{path}",
    headers=_HEADERS,
    params=params,
    timeout=10,
    follow_redirects=True
)
        return r.status_code, (r.json() if r.content else {})
    except httpx.RequestError as e:
        return -1, {"message": f"Network error: {e}"}
 
 
def _friendly_error(status: int, data: dict, context: str) -> str:
    if status == 403 and "rate limit" in str(data.get("message", "")).lower():
        return (f"ERROR: GitHub API rate limit exceeded. {context} "
                f"Unauthenticated requests are capped at 60/hour — set the GITHUB_TOKEN "
                f"environment variable to raise this to 5,000/hour.")
    if status == 404:
        return f"ERROR: {context} not found on GitHub (404)."
    if status == -1:
        return f"ERROR: could not reach GitHub. {data.get('message')}"
    return f"ERROR: GitHub API returned status {status} for {context}: {data.get('message', 'unknown error')}"
 
 
# -- TOOLS (real, live actions) -------------------------------------------
 

@mcp.tool()
def who_am_i() -> str:
    """Test GitHub authentication."""

    response = httpx.get(
        "https://api.github.com/user",
        headers=_HEADERS
    )

    return response.text


@mcp.tool()
def search_repositories(query: str) -> str:
    """Search public GitHub repositories by keyword, topic, or language.
    Args: query e.g. 'machine learning language:python'
    """
    status, data = _get("/search/repositories", params={"q": query, "sort": "stars", "per_page": 5})
    if status != 200:
        return _friendly_error(status, data, f"search for '{query}'")
    items = data.get("items", [])
    if not items:
        return f"No repositories found for query: {query}"
    lines = [f"Top results for '{query}':"]
    for repo in items:
        lines.append(f"- {repo['full_name']} ⭐{repo['stargazers_count']} — {repo['description']}")
    return "\n".join(lines)
 
 
@mcp.tool()
def get_repo_details(owner: str, repo: str) -> str:
    """
    Get live details for a GitHub repository.

    Args:
        owner: GitHub username or organization.
        repo: Repository name.
    """

    status, data = _get(f"/repos/{owner}/{repo}")

    # Repository was not found or another GitHub error occurred
    if status != 200:
        return (
            f"REPOSITORY_EXISTS: false\n"
            f"OWNER: {owner}\n"
            f"REPOSITORY: {repo}\n"
            f"ERROR: "
            f"{data.get('message', 'Unknown GitHub error')}"
        )

    # Repository exists
    license_name = (
        (data.get("license") or {}).get(
            "name",
            "No license"
        )
    )

    return (
        f"REPOSITORY_EXISTS: true\n"
        f"OWNER: {data['owner']['login']}\n"
        f"REPOSITORY: {data['name']}\n"
        f"FULL_NAME: {data['full_name']}\n"
        f"DESCRIPTION: "
        f"{data.get('description') or 'No description'}\n"
        f"STARS: {data['stargazers_count']:,}\n"
        f"FORKS: {data['forks_count']:,}\n"
        f"LANGUAGE: {data.get('language') or 'N/A'}\n"
        f"LICENSE: {license_name}\n"
        f"OPEN_ISSUES: {data['open_issues_count']}\n"
        f"DEFAULT_BRANCH: {data['default_branch']}\n"
        f"URL: {data['html_url']}"
    )
 
 
@mcp.tool()
def list_open_issues(owner: str, repo: str, limit: int = 5) -> str:
    """List currently open issues for a GitHub repository.
    Args: owner, repo, limit (max issues to return, default 5)
    """
    status, data = _get(f"/repos/{owner}/{repo}/issues", params={"state": "open", "per_page": limit})
    if status != 200:
        return _friendly_error(status, data, f"{owner}/{repo} issues")
    if not data:
        return f"{owner}/{repo} currently has no open issues."
    lines = [f"Open issues for {owner}/{repo} (showing {len(data)}):"]
    for issue in data:
        labels = ", ".join(l["name"] for l in issue.get("labels", [])) or "no labels"
        lines.append(f"- #{issue['number']}: {issue['title']} [{labels}]")
    return "\n".join(lines)
 
 
@mcp.tool()
def list_contributors(owner: str, repo: str, limit: int = 5) -> str:
    """List the top contributors to a GitHub repository by commit count.
    Args: owner, repo, limit (max contributors to return, default 5)
    """
    status, data = _get(f"/repos/{owner}/{repo}/contributors", params={"per_page": limit})
    if status != 200:
        return _friendly_error(status, data, f"{owner}/{repo} contributors")
    if not data:
        return f"No contributor data available for {owner}/{repo}."
    lines = [f"Top contributors to {owner}/{repo}:"]
    for c in data:
        lines.append(f"- {c['login']}: {c['contributions']} commits")
    return "\n".join(lines)
 
 
@mcp.tool()
def get_latest_release(owner: str, repo: str) -> str:
    """Get the latest published release for a GitHub repository.
    Args: owner, repo
    """
    status, data = _get(f"/repos/{owner}/{repo}/releases/latest")
    if status != 200:
        return _friendly_error(status, data, f"{owner}/{repo} latest release")
    return (f"Latest release of {owner}/{repo}: {data['tag_name']} ({data.get('name', '')})\n"
            f"Published: {data['published_at']}\n"
            f"Notes: {(data.get('body') or 'No release notes.')[:300]}")
 
 
# -- RESOURCE (read-only data, addressed by URI) ---------------------------
 
@mcp.resource("github://repo/{owner}/{repo}/summary")
def repo_summary_resource(owner: str, repo: str) -> str:
    """A pre-formatted, read-only snapshot of a repository's key stats."""
    status, data = _get(f"/repos/{owner}/{repo}")
    if status != 200:
        return _friendly_error(status, data, f"{owner}/{repo}")
    return (f"REPO SUMMARY: {data['full_name']}\n"
            f"Stars: {data['stargazers_count']} | Forks: {data['forks_count']} | "
            f"Open Issues: {data['open_issues_count']}\n"
            f"Created: {data['created_at']} | Last push: {data['pushed_at']}")

 


# -- RESOURCE (write data, addressed by URI) ---------------------------
@mcp.tool()
def create_repository(
    name: str,
    description: str = "",
    private: bool = False
) -> str:
    """
    Create a new GitHub repository.
    """

    url = "https://api.github.com/user/repos"

    payload = {
        "name": name,
        "description": description,
        "private": private
    }

    response = httpx.post(
        url,
        headers=_HEADERS,
        json=payload
    )

    if response.status_code == 201:
        repo = response.json()

        return (
            f"Repository '{repo['name']}' created successfully.\n"
            f"URL: {repo['html_url']}"
        )

    return response.text


@mcp.tool()
def create_file(
    owner: str,
    repo: str,
    path: str,
    content: str,
    message: str = "Create file"
) -> str:
    """
    Create a new file in a GitHub repository.

    Args:
        owner: GitHub username or organization.
        repo: Repository name.
        path: File path, for example README.md or .gitignore.
        content: File content.
        message: Git commit message.
    """

    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

    # GitHub Contents API requires Base64 encoded file content
    encoded_content = base64.b64encode(
        content.encode("utf-8")
    ).decode("utf-8")

    payload = {
        "message": message,
        "content": encoded_content
    }

    try:
        response = httpx.put(
            url,
            headers=_HEADERS,
            json=payload,
            timeout=10,
            follow_redirects=True
        )

    except httpx.RequestError as e:
        return f"ERROR: could not reach GitHub while creating '{path}': {e}"

    if response.status_code == 201:
        data = response.json()

        commit = data.get("commit", {})
        commit_sha = commit.get("sha", "unknown")

        return (
            f"File '{path}' created successfully in "
            f"{owner}/{repo}.\n"
            f"Commit: {commit_sha}"
        )

    try:
        error_data = response.json()
    except Exception:
        error_data = {
            "message": response.text
        }

    return _friendly_error(
        response.status_code,
        error_data,
        f"creating file '{path}' in {owner}/{repo}"
    )




@mcp.tool()
def delete_file(
    owner: str,
    repo: str,
    path: str,
    message: str = "Delete file"
) -> str:
    """
    Delete an existing file from a GitHub repository.

    Args:
        owner: GitHub username or organization.
        repo: Repository name.
        path: Path of the file to delete.
        message: Git commit message.
    """

    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

    # =========================================================
    # STEP 1: Get the existing file information
    # =========================================================

    try:
        get_response = httpx.get(
            url,
            headers=_HEADERS,
            timeout=10,
            follow_redirects=True
        )

    except httpx.RequestError as e:
        return (
            f"ERROR: could not reach GitHub while finding "
            f"'{path}': {e}"
        )

    # =========================================================
    # STEP 2: Check whether the file exists
    # =========================================================

    if get_response.status_code != 200:

        try:
            error_data = get_response.json()
        except Exception:
            error_data = {
                "message": get_response.text
            }

        return _friendly_error(
            get_response.status_code,
            error_data,
            f"finding file '{path}' in {owner}/{repo}"
        )

    # =========================================================
    # STEP 3: Get the file SHA
    # =========================================================

    file_data = get_response.json()

    file_sha = file_data.get("sha")

    if not file_sha:
        return (
            f"ERROR: GitHub did not return a SHA for "
            f"'{path}'."
        )

    # =========================================================
    # STEP 4: Prepare delete request
    # =========================================================

    payload = {
        "message": message,
        "sha": file_sha
    }

    # =========================================================
    # STEP 5: Delete the file
    # =========================================================

    try:
        delete_response = httpx.request(
            "DELETE",
            url,
            headers=_HEADERS,
            json=payload,
            timeout=10,
            follow_redirects=True
        )

    except httpx.RequestError as e:
        return (
            f"ERROR: could not reach GitHub while deleting "
            f"'{path}': {e}"
        )

    # =========================================================
    # STEP 6: Check result
    # =========================================================

    if delete_response.status_code == 200:

        data = delete_response.json()

        commit = data.get("commit", {})
        commit_sha = commit.get("sha", "unknown")

        return (
            f"File '{path}' deleted successfully from "
            f"{owner}/{repo}.\n"
            f"Commit: {commit_sha}"
        )

    # =========================================================
    # STEP 7: Handle GitHub error
    # =========================================================

    try:
        error_data = delete_response.json()
    except Exception:
        error_data = {
            "message": delete_response.text
        }

    return _friendly_error(
        delete_response.status_code,
        error_data,
        f"deleting file '{path}' from {owner}/{repo}"
    )


@mcp.tool()
def list_my_repositories(limit: int = 10) -> str:
    """
    List repositories owned by the authenticated GitHub user.
    search repositories of the authenticated user when requested.
    if request is made one repository, return the details(visibility, description) of that repository only.

      if no repositories are found, return a message indicating that the user has no repositories.

    """

    url = "https://api.github.com/user/repos"

    params = {
        "per_page": limit,
        "sort": "updated",
        "direction": "desc"
    }

    response = httpx.get(
        url,
        headers=_HEADERS,
        params=params
    )

    if response.status_code != 200:
        return _friendly_error(
            response.status_code,
            response.json(),
            "listing your repositories"
        )

    repositories = response.json()

    if not repositories:
        return "You don't have any repositories."

    lines = ["Your GitHub repositories:"]

    for repo in repositories:
        visibility = "Private" if repo["private"] else "Public"

        lines.append(
            f"- {repo['full_name']} "
            f"({visibility}) — {repo.get('description') or 'No description'}"
        )

    return "\n".join(lines)


@mcp.tool()
def list_repository_files(
    owner: str,
    repo: str,
    path: str = ""
) -> str:
    """
    List files and folders in a GitHub repository owned by
    the authenticated GitHub user.

    Args:
        owner: Repository owner.
        repo: Repository name, for example Food.
        path: Optional folder path inside the repository.
              Leave empty to list the repository root.
    """

    # ---------------------------------------------------------
    # Get authenticated GitHub username
    # ---------------------------------------------------------

   # try:
   #     owner = _get_authenticated_user()
   # except RuntimeError as e:
   #     return f"ERROR: {e}"

    # ---------------------------------------------------------
    # GitHub Contents API
    # ---------------------------------------------------------

    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

    try:
        response = httpx.get(
            url,
            headers=_HEADERS,
            timeout=10,
            follow_redirects=True
        )

    except httpx.RequestError as e:
        return (
            f"ERROR: could not reach GitHub while "
            f"listing files in {owner}/{repo}: {e}"
        )

    # ---------------------------------------------------------
    # Handle GitHub errors
    # ---------------------------------------------------------

    if response.status_code != 200:

        try:
            error_data = response.json()
        except Exception:
            error_data = {
                "message": response.text
            }

        return _friendly_error(
            response.status_code,
            error_data,
            f"listing files in {owner}/{repo}/{path}"
        )

    # ---------------------------------------------------------
    # Parse response
    # ---------------------------------------------------------

    data = response.json()

    # If the path points to a file
    if isinstance(data, dict):

        if data.get("type") == "file":
            return (
                f"'{path}' is a file, not a directory.\n"
                f"URL: {data.get('html_url', 'N/A')}"
            )

        return "Unexpected response from GitHub."

    if not data:
        location = path if path else "repository root"

        return (
            f"No files found in {owner}/{repo} "
            f"({location})."
        )

    # ---------------------------------------------------------
    # Format result
    # ---------------------------------------------------------

    location = path if path else "repository root"

    lines = [
        f"Files in {owner}/{repo} ({location}):"
    ]

    directories = [
        item for item in data
        if item.get("type") == "dir"
    ]

    files = [
        item for item in data
        if item.get("type") == "file"
    ]

    for item in directories:
        lines.append(
            f"📁 {item['name']}/"
        )

    for item in files:
        lines.append(
            f"📄 {item['name']}"
        )

    return "\n".join(lines)


@mcp.tool()
def get_file_details(
    owner: str,
    repo: str,
    path: str
) -> str:
    """
    Get detailed information and content of a single file
    in a GitHub repository.

    if user asked for the repository is present or not.give output with correct deatils.

    Args:
        owner: GitHub username or organization.
        repo: Repository name.
        path: Path to the file, for example README.md
              or src/app.py.
    """

    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

    try:
        response = httpx.get(
            url,
            headers=_HEADERS,
            timeout=10,
            follow_redirects=True
        )

    except httpx.RequestError as e:
        return (
            f"ERROR: could not reach GitHub while getting "
            f"'{path}': {e}"
        )

    # ---------------------------------------------------------
    # Handle GitHub errors
    # ---------------------------------------------------------

    if response.status_code != 200:

        try:
            error_data = response.json()
        except Exception:
            error_data = {
                "message": response.text
            }

        return _friendly_error(
            response.status_code,
            error_data,
            f"getting file '{path}' from {owner}/{repo}"
        )

    data = response.json()

    # ---------------------------------------------------------
    # Make sure the requested path is actually a file
    # ---------------------------------------------------------

    if data.get("type") != "file":
        return (
            f"'{path}' is not a file. "
            f"It is a {data.get('type', 'unknown')}."
        )

    # ---------------------------------------------------------
    # Decode file content
    # ---------------------------------------------------------

    encoded_content = data.get("content", "")

    try:
        # GitHub sometimes includes newline characters
        # in the Base64 response.
        encoded_content = encoded_content.replace("\n", "")

        decoded_content = base64.b64decode(
            encoded_content
        ).decode("utf-8")

    except (ValueError, UnicodeDecodeError):
        decoded_content = (
            "[File content could not be decoded as UTF-8.]"
        )

    # ---------------------------------------------------------
    # Return file details
    # ---------------------------------------------------------

    return (
        f"FILE DETAILS\n"
        f"------------------------------\n"
        f"Repository : {owner}/{repo}\n"
        f"Name       : {data.get('name', 'N/A')}\n"
        f"Path       : {data.get('path', 'N/A')}\n"
        f"Type       : {data.get('type', 'N/A')}\n"
        f"Size       : {data.get('size', 0)} bytes\n"
        f"SHA        : {data.get('sha', 'N/A')}\n"
        f"URL        : {data.get('html_url', 'N/A')}\n"
        f"------------------------------\n"
        f"CONTENT\n"
        f"------------------------------\n"
        f"{decoded_content}"
    )





@mcp.tool()
def edit_repository(
    owner: str,
    repo: str,
    description: str | None = None,
    private: bool | None = None,
    homepage: str | None = None
) -> str:
    """
    Edit GitHub repository settings such as Name,
    description, visibility, or homepage.
    """

    url = f"https://api.github.com/repos/{owner}/{repo}"

    payload = {}

    if description is not None:
        payload["description"] = description

    if private is not None:
        payload["private"] = private

    if homepage is not None:
        payload["homepage"] = homepage

    if not payload:
        return "No changes were provided."

    response = httpx.patch(
        url,
        headers=_HEADERS,
        json=payload
    )

    if response.status_code != 200:
        return _friendly_error(
            response.status_code,
            response.json(),
            f"editing {owner}/{repo}"
        )

    data = response.json()

    return (
        f"Repository '{data['full_name']}' updated successfully.\n"
        f"URL: {data['html_url']}"
    )


@mcp.tool()
def delete_repository(owner: str, repo: str) -> str:
    """
    Permanently delete a GitHub repository.
    This is an irreversible action.
    """

    url = f"https://api.github.com/repos/{owner}/{repo}"

    response = httpx.delete(
        url,
        headers=_HEADERS
    )

    if response.status_code == 204:
        return f"Repository '{owner}/{repo}' deleted successfully."

    try:
        data = response.json()
    except Exception:
        data = {"message": response.text}

    return _friendly_error(
        response.status_code,
        data,
        f"deleting {owner}/{repo}"
    )


@mcp.tool()
def publish_project(
    project_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Initial commit"
) -> str:
    """
    Publish a local project to GitHub.

    This will:
    1. Create the GitHub repository if it does not exist.
    2. Initialize Git in the local project if needed.
    3. Add the project files.
    4. Create a commit.
    5. Set the main branch.
    6. Add the GitHub repository as origin.
    7. Push the project to GitHub.

    Args:
        project_path: Local path of the project.
        repo_name: Name of the GitHub repository.
        private: Whether the GitHub repository should be private.
        commit_message: Git commit message.
    """

    # =========================================================
    # STEP 1: Validate project path
    # =========================================================

    project = Path(project_path).expanduser().resolve()

    if not project.exists():
        return (
            f"ERROR: Project path does not exist:\n"
            f"{project}"
        )

    if not project.is_dir():
        return (
            f"ERROR: Project path is not a directory:\n"
            f"{project}"
        )

    # =========================================================
    # STEP 2: Check GitHub authentication
    # =========================================================

    if not GITHUB_TOKEN:
        return (
            "ERROR: GITHUB_TOKEN is not configured."
        )

    # =========================================================
    # STEP 3: Get authenticated GitHub user
    # =========================================================

    try:
        user_response = httpx.get(
            f"{GITHUB_API}/user",
            headers=_HEADERS,
            timeout=10,
            follow_redirects=True
        )

    except httpx.RequestError as e:
        return (
            f"ERROR: Could not connect to GitHub: {e}"
        )

    if user_response.status_code != 200:

        try:
            error_data = user_response.json()
        except Exception:
            error_data = {
                "message": user_response.text
            }

        return _friendly_error(
            user_response.status_code,
            error_data,
            "getting authenticated GitHub user"
        )

    user_data = user_response.json()

    owner = user_data.get("login")

    if not owner:
        return (
            "ERROR: Could not determine authenticated "
            "GitHub username."
        )

    # =========================================================
    # STEP 4: Check whether repository already exists
    # =========================================================

    repo_url = f"{GITHUB_API}/repos/{owner}/{repo_name}"

    try:
        repo_check = httpx.get(
            repo_url,
            headers=_HEADERS,
            timeout=10,
            follow_redirects=True
        )

    except httpx.RequestError as e:
        return (
            f"ERROR: Could not check GitHub repository: {e}"
        )

    # =========================================================
    # Repository doesn't exist → create it
    # =========================================================

    if repo_check.status_code == 404:

        create_response = httpx.post(
            f"{GITHUB_API}/user/repos",
            headers=_HEADERS,
            json={
                "name": repo_name,
                "description": f"Published from {project.name}",
                "private": private,
                "auto_init": False
            },
            timeout=10,
            follow_redirects=True
        )

        if create_response.status_code != 201:

            try:
                error_data = create_response.json()
            except Exception:
                error_data = {
                    "message": create_response.text
                }

            return _friendly_error(
                create_response.status_code,
                error_data,
                f"creating repository {repo_name}"
            )

        repo_data = create_response.json()

    # =========================================================
    # Repository already exists
    # =========================================================

    elif repo_check.status_code == 200:

        repo_data = repo_check.json()

    else:

        try:
            error_data = repo_check.json()
        except Exception:
            error_data = {
                "message": repo_check.text
            }

        return _friendly_error(
            repo_check.status_code,
            error_data,
            f"checking repository {owner}/{repo_name}"
        )

    github_url = repo_data["html_url"]

    # =========================================================
    # Helper for running Git commands
    # =========================================================

    def run_git(*args):

        result = subprocess.run(
            ["git", *args],
            cwd=str(project),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown Git error"
            )

        return result.stdout.strip()

    # =========================================================
    # STEP 5: Initialize Git if necessary
    # =========================================================

    git_directory = project / ".git"

    try:

        if not git_directory.exists():

            run_git("init")

        # =====================================================
        # STEP 6: Configure branch
        # =====================================================

        run_git(
            "branch",
            "-M",
            "main"
        )

        # =====================================================
        # STEP 7: Configure Git identity
        # =====================================================

        try:
            current_name = run_git(
                "config",
                "--get",
                "user.name"
            )
        except RuntimeError:
            current_name = ""

        if not current_name:
            run_git(
                "config",
                "user.name",
                owner
            )

        try:
            current_email = run_git(
                "config",
                "--get",
                "user.email"
            )
        except RuntimeError:
            current_email = ""

        if not current_email:

            run_git(
                "config",
                "user.email",
                f"{owner}@users.noreply.github.com"
            )

        # =====================================================
        # STEP 8: Add project files
        # =====================================================

        run_git(
            "add",
            "."
        )

        # =====================================================
        # STEP 9: Check whether there is anything to commit
        # =====================================================

        status_result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain"
            ],
            cwd=str(project),
            capture_output=True,
            text=True
        )

        if status_result.returncode != 0:

            raise RuntimeError(
                status_result.stderr.strip()
            )

        if status_result.stdout.strip():

            run_git(
                "commit",
                "-m",
                commit_message
            )

        # =====================================================
        # STEP 10: Check existing remote
        # =====================================================

        try:
            existing_remote = run_git(
                "remote",
                "get-url",
                "origin"
            )
        except RuntimeError:
            existing_remote = ""

        expected_remote = github_url + ".git"

        # =====================================================
        # Add remote if it doesn't exist
        # =====================================================

        if not existing_remote:

            run_git(
                "remote",
                "add",
                "origin",
                expected_remote
            )

        # =====================================================
        # Don't overwrite an unrelated remote
        # =====================================================

        elif "github.com" not in existing_remote:

            return (
                "ERROR: The project already has an 'origin' "
                "remote that is not a GitHub repository.\n"
                f"Existing remote: {existing_remote}\n"
                f"Expected GitHub repository: {github_url}"
            )

        # =====================================================
        # STEP 11: Push to GitHub
        # =====================================================

        git_env = os.environ.copy()

        # Use Git's temporary configuration through environment
        # variables so the GitHub token is NOT stored in the
        # repository's remote URL.
        git_env["GIT_CONFIG_COUNT"] = "1"
        git_env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
        git_env["GIT_CONFIG_VALUE_0"] = (
            f"Authorization: Bearer {GITHUB_TOKEN}"
        )

        push_result = subprocess.run(
            [
                "git",
                "push",
                "-u",
                "origin",
                "main"
            ],
            cwd=str(project),
            capture_output=True,
            text=True,
            env=git_env
        )

        if push_result.returncode != 0:

            return (
                "ERROR: GitHub repository was created, "
                "but pushing the project failed.\n\n"
                f"Repository: {github_url}\n\n"
                f"Git error:\n"
                f"{push_result.stderr.strip()}"
            )

    except FileNotFoundError:

        return (
            "ERROR: Git is not installed or is not available "
            "in PATH."
        )

    except RuntimeError as e:

        return (
            f"ERROR while publishing project:\n{e}"
        )

    # =========================================================
    # SUCCESS
    # =========================================================

    return (
        "PROJECT PUBLISHED SUCCESSFULLY\n"
        "----------------------------------------\n"
        f"Local project : {project}\n"
        f"GitHub owner  : {owner}\n"
        f"Repository    : {repo_name}\n"
        f"Visibility    : "
        f"{'Private' if private else 'Public'}\n"
        f"Branch        : main\n"
        f"GitHub URL    : {github_url}\n"
        "----------------------------------------\n"
        "The local project has been committed and pushed to GitHub."
    )



# -- PROMPT (reusable instruction template) --------------------------------
 
@mcp.prompt()
def issue_triage_prompt() -> str:
    """A structured workflow for triaging a GitHub issue."""
    return """
    Review this GitHub issue and:
    1. Classify it as exactly one of: Bug / Feature Request / Question / Documentation
    2. Assign a priority: Critical / High / Medium / Low, based on impact described
    3. Suggest which existing label(s) it should carry
    4. Write a one-sentence summary suitable for a triage dashboard
    Be concise — output should fit in 4 short lines.
    """
 
 
# -- RUN THE SERVER ---------------------------------------------------------
 
if __name__ == "__main__":
    import sys
 
    print("Starting DevPulse MCP Server...", file=sys.stderr)
    print("5 tools: search_repositories, get_repo_details, list_open_issues, "
          "list_contributors, get_latest_release", file=sys.stderr)
    print("1 resource: github://repo/{owner}/{repo}/summary", file=sys.stderr)
    print("1 prompt: issue_triage_prompt", file=sys.stderr)
    if not GITHUB_TOKEN:
        print("NOTE: no GITHUB_TOKEN set — limited to 60 unauthenticated requests/hour.", file=sys.stderr)
    print("Waiting for MCP client connections (Ctrl+C to stop)...\n", file=sys.stderr)
    mcp.run()
 
 #to run the server, use the command: `npx @modelcontextprotocol/inspector python MCP_server.py`'