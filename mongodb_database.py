import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "devpulse")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is missing in .env")

import certifi

client = MongoClient(
    MONGODB_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=20000
)

db = client[MONGODB_DATABASE]

users = db["users"]
conversations = db["conversations"]
repositories = db["repositories"]
repository_actions = db["repository_actions"]
activities = db["activities"]


def now():
    return datetime.now(timezone.utc)


def require_user(user_id: str):
    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")


def repo_id(owner: str, repo: str):
    if not owner or not repo:
        raise ValueError("owner and repo are required")
    return f"{owner.strip().lower()}/{repo.strip().lower()}"


def check_connection() -> bool:
    try:
        client.admin.command("ping")
        return True
    except PyMongoError as exc:
        print(f"MongoDB connection failed: {exc}")
        return False


def create_indexes():
    users.create_index([("user_id", ASCENDING)], unique=True)
    conversations.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])
    repositories.create_index([("user_id", ASCENDING), ("repo_id", ASCENDING)], unique=True)
    repository_actions.create_index([("user_id", ASCENDING), ("repo_id", ASCENDING), ("created_at", DESCENDING)])
    activities.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])


def save_user_profile(user_id: str, name: str, email: str, provider: str,
                      provider_user_id: str, extra: Optional[Dict[str, Any]] = None):
    require_user(user_id)
    profile = {
        "user_id": user_id,
        "name": name or "",
        "email": email or "",
        "provider": provider or "",
        "provider_user_id": provider_user_id or "",
        "extra": extra or {},
        "updated_at": now(),
    }
    users.update_one(
        {"user_id": user_id},
        {"$set": profile, "$setOnInsert": {"created_at": now()}},
        upsert=True,
    )
    return profile


def get_user_profile(user_id: str):
    require_user(user_id)
    return users.find_one({"user_id": user_id}, {"_id": 0})


def append_message(user_id: str, message: Dict[str, Any]):
    require_user(user_id)
    conversations.insert_one({
        "user_id": user_id,
        "message": message,
        "created_at": now(),
    })


def load_history(user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    require_user(user_id)
    limit = max(1, min(limit, 1000))
    cursor = conversations.find(
        {"user_id": user_id},
        {"_id": 0, "message": 1},
    ).sort("created_at", ASCENDING).limit(limit)
    return [item["message"] for item in cursor]


def save_history(user_id: str, history: List[Dict[str, Any]]):
    require_user(user_id)

    conversations.delete_many({
        "user_id": user_id
    })

    if history:
        documents = [
            {
                "user_id": user_id,
                "message": message,
                "created_at": now()
            }
            for message in history
        ]

        conversations.insert_many(documents)

        

def clear_history(user_id: str) -> int:
    require_user(user_id)
    return conversations.delete_many({"user_id": user_id}).deleted_count


def save_repository(user_id: str, owner: str, repo: str,
                    metadata: Optional[Dict[str, Any]] = None):
    require_user(user_id)
    metadata = metadata or {}
    rid = repo_id(owner, repo)

    document = {
        "user_id": user_id,
        "repo_id": rid,
        "owner": owner,
        "repo": repo,
        "full_name": metadata.get("full_name", f"{owner}/{repo}"),
        "description": metadata.get("description", ""),
        "language": metadata.get("language", ""),
        "default_branch": metadata.get("default_branch", "main"),
        "private": metadata.get("private", False),
        "url": metadata.get("url", ""),
        "stars": metadata.get("stars", 0),
        "forks": metadata.get("forks", 0),
        "open_issues": metadata.get("open_issues", 0),
        "last_seen": now(),
    }

    repositories.update_one(
        {"user_id": user_id, "repo_id": rid},
        {"$set": document, "$setOnInsert": {"created_at": now()}},
        upsert=True,
    )
    return document


def get_repository(user_id: str, owner: str, repo: str):
    require_user(user_id)
    return repositories.find_one(
        {"user_id": user_id, "repo_id": repo_id(owner, repo)},
        {"_id": 0},
    )


def list_repositories(user_id: str):
    require_user(user_id)
    return list(repositories.find(
        {"user_id": user_id},
        {"_id": 0},
    ).sort("last_seen", DESCENDING))


def delete_repository_memory(user_id: str, owner: str, repo: str):
    require_user(user_id)
    rid = repo_id(owner, repo)
    repo_result = repositories.delete_one({"user_id": user_id, "repo_id": rid})
    action_result = repository_actions.delete_many({"user_id": user_id, "repo_id": rid})
    return repo_result.deleted_count + action_result.deleted_count


def add_repository_action(user_id: str, owner: str, repo: str, action: str,
                          status: str, details: Optional[Dict[str, Any]] = None):
    require_user(user_id)
    event = {
        "user_id": user_id,
        "repo_id": repo_id(owner, repo),
        "owner": owner,
        "repo": repo,
        "action": action,
        "status": status,
        "details": details or {},
        "created_at": now(),
    }
    repository_actions.insert_one(event)
    return event


def get_repository_actions(user_id: str, owner: str, repo: str, limit: int = 20):
    require_user(user_id)
    limit = max(1, min(limit, 100))
    return list(repository_actions.find(
        {"user_id": user_id, "repo_id": repo_id(owner, repo)},
        {"_id": 0},
    ).sort("created_at", DESCENDING).limit(limit))


def add_activity(user_id: str, action: str, status: str,
                 details: Optional[Dict[str, Any]] = None):
    require_user(user_id)
    event = {
        "user_id": user_id,
        "action": action,
        "status": status,
        "details": details or {},
        "created_at": now(),
    }
    activities.insert_one(event)
    return event


def get_activity(user_id: str, limit: int = 20):
    require_user(user_id)
    limit = max(1, min(limit, 200))
    return list(activities.find(
        {"user_id": user_id},
        {"_id": 0},
    ).sort("created_at", DESCENDING).limit(limit))


def clear_user_data(user_id: str):
    require_user(user_id)
    return {
        "users": users.delete_many({"user_id": user_id}).deleted_count,
        "conversations": conversations.delete_many({"user_id": user_id}).deleted_count,
        "repositories": repositories.delete_many({"user_id": user_id}).deleted_count,
        "repository_actions": repository_actions.delete_many({"user_id": user_id}).deleted_count,
        "activities": activities.delete_many({"user_id": user_id}).deleted_count,
    }


if __name__ == "__main__":
    print("Connected:", check_connection())
    if check_connection():
        create_indexes()
        print("MongoDB indexes created.")
