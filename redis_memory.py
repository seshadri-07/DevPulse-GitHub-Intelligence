"""Redis Cloud persistence for DevPulse conversation history.

Each conversation message is stored as a separate JSON item in a Redis list.
This makes the data easier to inspect in Redis Cloud.
"""

import json
import os

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
REDIS_USER_ID = os.getenv("DEVPULSE_USER_ID", "default_user")

if not REDIS_URL:
    raise ValueError("REDIS_URL is missing from the .env file.")

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=10,
    socket_timeout=10,
)


def check_redis_connection() -> bool:
    try:
        redis_client.ping()
        print("Redis Cloud connected successfully!")
        return True
    except redis.RedisError as error:
        print(f"Redis Cloud connection failed: {error}")
        return False


def _history_key(user_id: str = REDIS_USER_ID) -> str:
    return f"devpulse:history:{user_id}"


def load_history(user_id: str = REDIS_USER_ID) -> list:
    """Load each message from a Redis list."""
    try:
        raw_messages = redis_client.lrange(
            _history_key(user_id),
            0,
            -1
        )

        history = []

        for raw_message in raw_messages:
            try:
                message = json.loads(raw_message)
                if isinstance(message, dict):
                    history.append(message)
            except json.JSONDecodeError:
                print("Warning: skipped invalid history item.")

        return history

    except redis.RedisError as error:
        print(f"Could not load history from Redis Cloud: {error}")
        return []


def save_history(history: list, user_id: str = REDIS_USER_ID) -> bool:
    """Replace the Redis list with clearly separated message entries."""
    try:
        key = _history_key(user_id)

        # Prevent duplicate messages when saving the complete history.
        pipeline = redis_client.pipeline()
        pipeline.delete(key)

        if history:
            encoded_messages = [
                json.dumps(message, ensure_ascii=False)
                for message in history
            ]
            pipeline.rpush(key, *encoded_messages)

        pipeline.execute()
        return True

    except (redis.RedisError, TypeError) as error:
        print(f"Could not save history to Redis Cloud: {error}")
        return False


def append_message(
    role: str,
    content: str,
    user_id: str = REDIS_USER_ID
) -> bool:
    """Append one readable message without rewriting all history."""
    try:
        message = {
            "role": role,
            "content": content
        }

        redis_client.rpush(
            _history_key(user_id),
            json.dumps(message, ensure_ascii=False)
        )

        return True

    except (redis.RedisError, TypeError) as error:
        print(f"Could not append message: {error}")
        return False


def clear_history(user_id: str = REDIS_USER_ID) -> bool:
    try:
        redis_client.delete(_history_key(user_id))
        return True
    except redis.RedisError as error:
        print(f"Could not clear history from Redis Cloud: {error}")
        return False


def print_history(user_id: str = REDIS_USER_ID) -> None:
    """Print conversation history in a readable format."""
    history = load_history(user_id)

    if not history:
        print("No conversation history found.")
        return

    print("\n========== DEVPULSE CONVERSATION HISTORY ==========")

    for index, message in enumerate(history, start=1):
        print(f"\nMessage {index}")
        print(f"Role: {message.get('role', 'unknown')}")
        print("Content:")

        content = message.get("content", "")

        if isinstance(content, str):
            print(content)
        else:
            print(json.dumps(content, indent=2, ensure_ascii=False))

        if message.get("tool_calls"):
            print("Tool Calls:")
            print(json.dumps(
                message["tool_calls"],
                indent=2,
                ensure_ascii=False
            ))

        if message.get("tool_call_id"):
            print(f"Tool Call ID: {message['tool_call_id']}")

    print("\n===================================================")


if __name__ == "__main__":
    if check_redis_connection():
        print_history()
