import os
from dotenv import load_dotenv
from mongodb_database import check_connection, create_indexes, save_user_profile, save_repository, add_repository_action, add_activity, list_repositories

load_dotenv()
user_id = os.getenv("DEVPULSE_USER_ID", "local_test_user")

if not check_connection():
    raise SystemExit("MongoDB connection failed")

create_indexes()
save_user_profile(user_id, "Test User", "test@example.com", "local", "local_001")
save_repository(user_id, "octocat", "hello-world", {"language": "Python"})
add_repository_action(user_id, "octocat", "hello-world", "get_repo_details", "success")
add_activity(user_id, "mongodb_test", "success")

print(list_repositories(user_id))
print("MongoDB test completed successfully.")