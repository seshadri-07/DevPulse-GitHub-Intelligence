from dotenv import load_dotenv
import os
import httpx

load_dotenv()

token = os.getenv("GITHUB_TOKEN")

print("Token:", token[:10] + "..." if token else None)
print("Length:", len(token) if token else 0)

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "DevPulse",
    "X-GitHub-Api-Version": "2022-11-28"
}

# Check authenticated user
r = httpx.get(
    "https://api.github.com/user",
    headers=headers
)

print("User Status:", r.status_code)
print("User:", r.text)
print("Scopes:", r.headers.get("X-OAuth-Scopes"))

print("\nCreating Repository...\n")

response = httpx.post(
    "https://api.github.com/user/repos",
    headers=headers,
    json={
        "name": "FoodTest123",
        "private": False,
        "auto_init": True
    }
)

print("Create Status:", response.status_code)
print(response.text)




"""
=============================================================
DevPulse — Agent Core
Model + Tools + Memory + Human-in-the-Loop (HITL)
 
Flow:
User
   ↓
LLM (Reason)
   ↓
Select Tool
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
 
 # single reAct agent 
import json
import asyncio
 
from model_client import get_client_and_model, check_connection
from mcp_bridge import mcp_session, discover_openai_tools, call_mcp_tool
 
 
SYSTEM_PROMPT = (
    "You are DevPulse, an assistant that answers questions about GitHub "
    "repositories using real, live tools. Always use a tool to look up facts "
    "instead of guessing. Remember repo names mentioned earlier in the "
    "conversation so the user doesn't have to repeat them."
)

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


 
 
class DevPulseAgent:
 
    def __init__(self):
        self.client, self.model, self.backend = get_client_and_model()
        self.history = []
 
    async def ask(self, user_input: str, max_loops: int = 4) -> str:
 
        self.history.append(
            {
                "role": "user",
                "content": user_input
            }
        )
 
        async with mcp_session() as session:
 
            # Discover MCP tools dynamically
            tools = await discover_openai_tools(session)
 
            for _ in range(max_loops):
 
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                ] + self.history
 
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
 
                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
 
                # No tool required → Final Answer
                if finish_reason != "tool_calls" or not msg.tool_calls:
 
                    self.history.append(
                        {
                            "role": "assistant",
                            "content": msg.content
                        }
                    )
 
                    return msg.content
 
                # Store assistant tool request
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



            
 
                # Execute each tool requested by the LLM
                for tool_call in msg.tool_calls:
 
                    args = json.loads(tool_call.function.arguments)
 
                    print("\n")
                    print("=" * 60)
                    print(" HUMAN-IN-THE-LOOP APPROVAL ")
                    print("=" * 60)
                    print(f"Tool Selected : {tool_call.function.name}")
                    print(f"Arguments     : {args}")
                    print("=" * 60)
 
                    approval = input(
                        "Approve tool execution? (y/n): "
                    ).strip().lower()
 
                    if approval != "y":
 
                        print("\n❌ Tool execution rejected by human.\n")
 
                        self.history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": "Tool execution denied by human reviewer."
                            }
                        )
 
                        continue
 
                    print("\n✅ Human approved.")
                    print("Executing MCP Tool...\n")
 
                    result_text = await call_mcp_tool(
                        session,
                        tool_call.function.name,
                        args
                    )
 
                    print("=" * 60)
                    print(" TOOL RESULT ")
                    print("=" * 60)
                    print(result_text)
                    print("=" * 60)
 
                    # Feed tool result back to LLM
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        }
                    )
 
        return "I wasn't able to resolve this within the allotted reasoning steps."
 
 
async def main():
 
    agent = DevPulseAgent()
 
    if not check_connection(
        agent.client,
        agent.model,
        agent.backend
    ):
        return
 
    print("\n")
    print("=" * 70)
    print(" DevPulse Agent with Human-in-the-Loop ")
    print("=" * 70)
    print(f"Model : {agent.model}")
    print("Type 'quit' to exit.")
    print("=" * 70)
 
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
 
 
if __name__ == "__main__":
    asyncio.run(main())