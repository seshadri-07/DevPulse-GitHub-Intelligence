"""
DevPulse Repository Memory
Stores repository metadata and repository-related actions in Redis Cloud.

Required environment variables:
    REDIS_URL
    DEVPULSE_USER_ID (fallback user ID for local testing)

Redis structures:
    devpulse:user:{user_id}:repos              -> Redis Set of repository IDs
    devpulse:user:{user_id}:repo:{repo_id}     -> Redis Hash with repository metadata
    devpulse:user:{user_id}:repo:{repo_id}:actions -> Redis List of JSON actions
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
REDIS_USER_ID = os.getenv("DEVPULSE_USER_ID", "default_user")

if not REDIS_URL:
    raise ValueError("REDIS_URL is missing in the .env file")

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_timeout=10,
    socket_connect_timeout=10,
    retry_on_timeout=True,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_repo_id(owner: str, repo: str) -> str:
    """Create a stable repository ID."""
    return f"{owner.strip().lower()}/{repo.strip().lower()}"


def _repo_key(user_id: str, repo_id: str) -> str:
    return f"devpulse:user:{user_id}:repo:{repo_id}"


def _repo_index_key(user_id: str) -> str:
    return f"devpulse:user:{user_id}:repos"


def _actions_key(user_id: str, repo_id: str) -> str:
    return f"{_repo_key(user_id, repo_id)}:actions"


def check_repo_memory_connection() -> bool:
    """Check whether Redis is reachable."""
    try:
        return bool(redis_client.ping())
    except redis.RedisError as exc:
        print(f"Redis repository memory connection failed: {exc}")
        return False


def save_repository(
    user_id: str,
    owner: str,
    repo: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create or update repository metadata for one authenticated user.

    The repository is scoped to user_id, so two users can store repositories
    with the same owner/repository name without mixing their data.
    """
    if not user_id:
        raise ValueError("user_id is required")
    if not owner or not repo:
        raise ValueError("owner and repo are required")

    metadata = metadata or {}
    repo_id = _safe_repo_id(owner, repo)
    key = _repo_key(user_id, repo_id)

    existing = get_repository(user_id, owner, repo) or {}

    record = {
        "repo_id": repo_id,
        "owner": owner,
        "repo": repo,
        "full_name": metadata.get("full_name", f"{owner}/{repo}"),
        "description": metadata.get("description", existing.get("description", "")),
        "language": metadata.get("language", existing.get("language", "")),
        "default_branch": metadata.get(
            "default_branch",
            existing.get("default_branch", "main"),
        ),
        "private": str(metadata.get("private", existing.get("private", False))).lower(),
        "url": metadata.get("url", existing.get("url", "")),
        "stars": str(metadata.get("stars", existing.get("stars", 0))),
        "forks": str(metadata.get("forks", existing.get("forks", 0))),
        "open_issues": str(
            metadata.get("open_issues", existing.get("open_issues", 0))
        ),
        "last_seen": _now(),
    }

    with redis_client.pipeline(transaction=True) as pipe:
        pipe.hset(key, mapping=record)
        pipe.sadd(_repo_index_key(user_id), repo_id)
        pipe.execute()

    return record


def get_repository(
    user_id: str,
    owner: str,
    repo: str,
) -> Optional[Dict[str, Any]]:
    """Get one repository's saved metadata."""
    repo_id = _safe_repo_id(owner, repo)
    data = redis_client.hgetall(_repo_key(user_id, repo_id))
    return data or None


def list_repositories(user_id: str) -> List[Dict[str, Any]]:
    """Return all repositories saved for a user."""
    repo_ids = sorted(redis_client.smembers(_repo_index_key(user_id)))
    repositories: List[Dict[str, Any]] = []

    for repo_id in repo_ids:
        data = redis_client.hgetall(_repo_key(user_id, repo_id))
        if data:
            repositories.append(data)

    return repositories


def delete_repository_memory(
    user_id: str,
    owner: str,
    repo: str,
) -> bool:
    """Delete repository metadata and its saved action history."""
    repo_id = _safe_repo_id(owner, repo)

    with redis_client.pipeline(transaction=True) as pipe:
        pipe.delete(_repo_key(user_id, repo_id))
        pipe.delete(_actions_key(user_id, repo_id))
        pipe.srem(_repo_index_key(user_id), repo_id)
        results = pipe.execute()

    return bool(results[0] or results[1] or results[2])


def add_repository_action(
    user_id: str,
    owner: str,
    repo: str,
    action: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save one action performed or attempted on a repository."""
    if not action:
        raise ValueError("action is required")

    repo_id = _safe_repo_id(owner, repo)
    event = {
        "timestamp": _now(),
        "repo_id": repo_id,
        "owner": owner,
        "repo": repo,
        "action": action,
        "status": status,
        "details": details or {},
    }

    redis_client.rpush(_actions_key(user_id, repo_id), json.dumps(event))
    redis_client.ltrim(_actions_key(user_id, repo_id), -100, -1)
    return event


def get_repository_actions(
    user_id: str,
    owner: str,
    repo: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Get the most recent repository actions."""
    repo_id = _safe_repo_id(owner, repo)
    raw_actions = redis_client.lrange(
        _actions_key(user_id, repo_id),
        max(-limit, -100),
        -1,
    )

    actions: List[Dict[str, Any]] = []
    for item in raw_actions:
        try:
            actions.append(json.loads(item))
        except json.JSONDecodeError:
            actions.append({"raw": item})

    return actions


def clear_all_repository_memory(user_id: str) -> int:
    """Delete all repository memory belonging to one user."""
    repo_ids = list(redis_client.smembers(_repo_index_key(user_id)))
    keys = [_repo_index_key(user_id)]

    for repo_id in repo_ids:
        keys.append(_repo_key(user_id, repo_id))
        keys.append(_actions_key(user_id, repo_id))

    if not keys:
        return 0

    return int(redis_client.delete(*keys))


def print_repository_memory(user_id: str) -> None:
    """Print repository memory in a readable format for debugging."""
    repositories = list_repositories(user_id)

    if not repositories:
        print("No repository memory found.")
        return

    for repository in repositories:
        print("\n" + "=" * 60)
        print(f"Repository: {repository.get('full_name')}")
        print(f"Description: {repository.get('description', '')}")
        print(f"Language: {repository.get('language', '')}")
        print(f"Default branch: {repository.get('default_branch', '')}")
        print(f"URL: {repository.get('url', '')}")
        print(f"Last seen: {repository.get('last_seen', '')}")

        owner = repository.get("owner", "")
        repo = repository.get("repo", "")
        actions = get_repository_actions(user_id, owner, repo)

        if actions:
            print("Recent actions:")
            for item in actions:
                print(
                    f"  - {item.get('timestamp')} | "
                    f"{item.get('action')} | {item.get('status')}"
                )
        else:
            print("Recent actions: none")


if __name__ == "__main__":
    if check_repo_memory_connection():
        print(f"Connected to Redis for user: {REDIS_USER_ID}")
        print_repository_memory(REDIS_USER_ID)
