

 
import os
import json
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
 
SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")
 
 
@asynccontextmanager
async def mcp_session():
    """Launch mcp_server.py as a subprocess and yield a ready ClientSession.
 
    NOTE: env=os.environ.copy() is required — the MCP stdio transport does
    NOT inherit the parent process's environment by default, so without this
    the server subprocess would never see GITHUB_TOKEN.
    """
    server_params = StdioServerParameters(
        # Keep the MCP server in the same virtual environment as Streamlit.
        command=sys.executable, args=[SERVER_PATH], env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session
 
 
async def discover_openai_tools(session: ClientSession) -> list[dict]:
    """Real MCP list_tools(), converted into OpenAI/Groq function-calling schema.
 
    MCP's own tool.inputSchema is already JSON Schema, so this is a thin
    reshape rather than a hand-written schema — proving the tool definitions
    really do come from the live server, not from a hard-coded list here.
    """
    tools_result = await session.list_tools()
    openai_tools = []
    for t in tools_result.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        })
    return openai_tools
 
 
async def call_mcp_tool(session: ClientSession, name: str, arguments: dict) -> str:
    """Real call_tool() against the live MCP server, flattened to plain text."""
    result = await session.call_tool(name, arguments=arguments)
    texts = [item.text for item in result.content if hasattr(item, "text")]
    return "\n".join(texts) if texts else "(empty result)"
 
 
