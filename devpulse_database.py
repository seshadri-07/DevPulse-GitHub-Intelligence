import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis
from dotenv import load_dotenv


load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")
DEFAULT_USER_ID = os.getenv("DEVPULSE_USER_ID", "local_test_user")

if not REDIS_URL:
    raise ValueError("REDIS_URL is missing in .env")

db = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=10)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_user(user_id: str):
    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")


def repo_id(owner: str, repo: str) -> str:
    if not owner or not repo:
        raise ValueError("owner and repo are required")
    return f"{owner.strip().lower()}/{repo.strip().lower()}"


def key(*parts: str) -> str:
    return ":".join(parts)


def user_key(user_id): return key("devpulse", "user", user_id)
def history_key(user_id): return key("devpulse", "history", user_id)
def repo_index_key(user_id): return key("devpulse", "user", user_id, "repos")
def repo_key(user_id, rid): return key("devpulse", "user", user_id, "repo", rid)
def repo_actions_key(user_id, rid): return key(repo_key(user_id, rid), "actions")
def activity_key(user_id): return key("devpulse", "user", user_id, "actions")


def check_connection() -> bool:
    try:
        return bool(db.ping())
    except redis.RedisError as exc:
        print(f"Redis connection failed: {exc}")
        return False


def save_user_profile(user_id: str, name: str, email: str, provider: str,
                      provider_user_id: str, extra: Optional[Dict[str, Any]] = None):
    require_user(user_id)
    profile = {
        "user_id": user_id, "name": name or "", "email": email or "",
        "provider": provider or "", "provider_user_id": provider_user_id or "",
        "updated_at": now(), "extra": json.dumps(extra or {})
    }
    db.hset(user_key(user_id), mapping=profile)
    return profile


def get_user_profile(user_id: str):
    require_user(user_id)
    data = db.hgetall(user_key(user_id))
    if data and "extra" in data:
        try: data["extra"] = json.loads(data["extra"])
        except json.JSONDecodeError: pass
    return data or None


def load_history(user_id: str) -> List[Dict[str, Any]]:
    require_user(user_id)
    result = []
    for item in db.lrange(history_key(user_id), 0, -1):
        try: result.append(json.loads(item))
        except json.JSONDecodeError: result.append({"content": item})
    return result


def append_message(user_id: str, message: Dict[str, Any]):
    require_user(user_id)
    db.rpush(history_key(user_id), json.dumps(message))


def save_history(user_id: str, history: List[Dict[str, Any]]):
    require_user(user_id)
    with db.pipeline(transaction=True) as pipe:
        pipe.delete(history_key(user_id))
        for message in history:
            pipe.rpush(history_key(user_id), json.dumps(message))
        pipe.execute()


def clear_history(user_id: str) -> bool:
    require_user(user_id)
    return bool(db.delete(history_key(user_id)))


def save_repository(user_id: str, owner: str, repo: str,
                    metadata: Optional[Dict[str, Any]] = None):
    require_user(user_id)
    metadata = metadata or {}
    rid = repo_id(owner, repo)
    old = get_repository(user_id, owner, repo) or {}
    record = {
        "repo_id": rid, "owner": owner, "repo": repo,
        "full_name": metadata.get("full_name", f"{owner}/{repo}"),
        "description": metadata.get("description", old.get("description", "")),
        "language": metadata.get("language", old.get("language", "")),
        "default_branch": metadata.get("default_branch", old.get("default_branch", "main")),
        "private": str(metadata.get("private", old.get("private", False))).lower(),
        "url": metadata.get("url", old.get("url", "")),
        "stars": str(metadata.get("stars", old.get("stars", 0))),
        "forks": str(metadata.get("forks", old.get("forks", 0))),
        "open_issues": str(metadata.get("open_issues", old.get("open_issues", 0))),
        "last_seen": now()
    }
    with db.pipeline(transaction=True) as pipe:
        pipe.hset(repo_key(user_id, rid), mapping=record)
        pipe.sadd(repo_index_key(user_id), rid)
        pipe.execute()
    return record


def get_repository(user_id: str, owner: str, repo: str):
    require_user(user_id)
    data = db.hgetall(repo_key(user_id, repo_id(owner, repo)))
    return data or None


def list_repositories(user_id: str):
    require_user(user_id)
    result = []
    for rid in sorted(db.smembers(repo_index_key(user_id))):
        data = db.hgetall(repo_key(user_id, rid))
        if data: result.append(data)
    return result


def delete_repository_memory(user_id: str, owner: str, repo: str) -> bool:
    require_user(user_id)
    rid = repo_id(owner, repo)
    with db.pipeline(transaction=True) as pipe:
        pipe.delete(repo_key(user_id, rid))
        pipe.delete(repo_actions_key(user_id, rid))
        pipe.srem(repo_index_key(user_id), rid)
        result = pipe.execute()
    return bool(any(result))


def add_repository_action(user_id: str, owner: str, repo: str, action: str,
                          status: str, details: Optional[Dict[str, Any]] = None):
    require_user(user_id)
    rid = repo_id(owner, repo)
    event = {"timestamp": now(), "repo_id": rid, "owner": owner, "repo": repo,
             "action": action, "status": status, "details": details or {}}
    rkey = repo_actions_key(user_id, rid)
    with db.pipeline(transaction=True) as pipe:
        pipe.rpush(rkey, json.dumps(event))
        pipe.ltrim(rkey, -100, -1)
        pipe.execute()
    return event


def get_repository_actions(user_id: str, owner: str, repo: str, limit: int = 20):
    require_user(user_id)
    limit = max(1, min(limit, 100))
    raw = db.lrange(repo_actions_key(user_id, repo_id(owner, repo)), -limit, -1)
    return [json.loads(item) for item in raw]


def add_activity(user_id: str, action: str, status: str,
                 details: Optional[Dict[str, Any]] = None):
    require_user(user_id)
    event = {"timestamp": now(), "action": action, "status": status,
             "details": details or {}}
    akey = activity_key(user_id)
    with db.pipeline(transaction=True) as pipe:
        pipe.rpush(akey, json.dumps(event))
        pipe.ltrim(akey, -200, -1)
        pipe.execute()
    return event


def get_activity(user_id: str, limit: int = 20):
    require_user(user_id)
    raw = db.lrange(activity_key(user_id), -max(1, min(limit, 200)), -1)
    return [json.loads(item) for item in raw]


def clear_user_data(user_id: str) -> int:
    require_user(user_id)
    ids = list(db.smembers(repo_index_key(user_id)))
    keys = [user_key(user_id), history_key(user_id), repo_index_key(user_id), activity_key(user_id)]
    for rid in ids:
        keys += [repo_key(user_id, rid), repo_actions_key(user_id, rid)]
    return int(db.delete(*keys))


if __name__ == "__main__":
    print("Connected:", check_connection())
    print("User:", DEFAULT_USER_ID)
    print("Repositories:", json.dumps(list_repositories(DEFAULT_USER_ID), indent=2))
