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
import os

from model_client import get_client_and_model, check_connection
from mcp_bridge import mcp_session, discover_openai_tools, call_mcp_tool

from devpulse_database import (
   load_history,
   save_history,
   append_message,
   save_repository,
    add_repository_action,
   list_repositories,
)
  
##from mongodb_database import (
  ##  check_connection as mongo_check_connection,
    ##load_history,
##    save_history,  clear_history,save_repository, add_repository_action, list_repositories,
##)


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
    def __init__(self, user_id: str):
        self.client, self.model, self.backend = get_client_and_model()
        self.user_id = user_id

        # Restore previous conversation history from MongoDB.
        try:
            if check_connection():
                self.history = load_history(self.user_id)
                print(f"Loaded {len(self.history)} messages from MongoDB.")
            else:
                self.history = []
                print("MongoDB unavailable. Starting with empty in-memory history.")
        except Exception as error:
            self.history = []
            print(f"MongoDB history could not be loaded: {error}")

    def _save_history(self) -> None:
        """Persist the current history without stopping the agent if MongoDB fails."""
        try:
            # unified database signature: save_history(user_id, history)
            save_history(self.user_id, self.history)
        except Exception as error:
            print(f"Warning: could not save history to MongoDB: {error}")

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
        self._save_history()

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
                    self._save_history()

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
                        self._save_history()

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
                            self._save_history()

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
                    self._save_history()

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

    agent = DevPulseAgent(user_id=os.getenv("DEVPULSE_USER_ID", "local_test_user"))

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