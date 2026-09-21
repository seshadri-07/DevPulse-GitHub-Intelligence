"""
=============================================================
DevPulse — LangGraph Multi-Agent Version
Router node classifies intent -> conditional edges dispatch to
one of 3 specialist nodes -> each calls the REAL MCP server.
 
=============================================================
"""
 
import os
import json
import asyncio
from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from model_client import get_client_and_model, check_connection
from mcp_bridge import mcp_session, discover_openai_tools, call_mcp_tool
 
client, MODEL, BACKEND = get_client_and_model()
 
 
class DevPulseState(TypedDict):
    query: str
    history: list           # short-term memory across turns
    category: str            # repo_info | issue_triage | release_notes
    tool_calls_made: list
    answer: str
 
 
# -- ROUTER NODE: classify intent --------------------------------------------
 
def classify_intent(state: DevPulseState) -> DevPulseState:
    prompt = (
        "Classify this GitHub-related query into exactly one category:\n"
        "repo_info, issue_triage, release_notes\n\n"
        "repo_info = general repo stats, search, contributors\n"
        "issue_triage = anything about open issues\n"
        "release_notes = anything about latest release/version\n\n"
        f"Query: {state['query']}\n\nReply with ONLY one of the three words."
    )
    result = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    category = result.choices[0].message.content.strip().lower()
    valid = {"repo_info", "issue_triage", "release_notes"}
    state["category"] = category if category in valid else "repo_info"
    print(f"  [router] classified as: {state['category']}")
    return state
 
 
def route(state: DevPulseState) -> str:
    return state["category"]
 
 
# -- SHARED SPECIALIST LOGIC: each node runs its own scoped ReAct mini-loop ---
 
async def _run_specialist(state: DevPulseState, persona: str, allowed_tools: set[str]) -> DevPulseState:
    """Every specialist shares one real MCP session but is restricted to its
    own subset of tools — this is what makes it a 'specialist' rather than
    just three copies of the same general agent."""
    async with mcp_session() as session:
        all_tools = await discover_openai_tools(session)
        scoped_tools = [t for t in all_tools if t["function"]["name"] in allowed_tools]
 
        messages = [{"role": "system", "content": persona}] + state["history"] + [
            {"role": "user", "content": state["query"]}
        ]
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=scoped_tools, tool_choice="auto"
        )
        msg = response.choices[0].message
 
        if response.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"  [{persona.split()[1]}] {tc.function.name}({args})")
                tool_result = await call_mcp_tool(session, tc.function.name, args)
                state["tool_calls_made"].append({"tool": tc.function.name, "result": tool_result})
            # Second pass: synthesize the final answer from real tool result(s)
            follow_up = messages + [
                {"role": "assistant", "content": msg.content, "tool_calls": [tc.model_dump() for tc in msg.tool_calls]}
            ] + [
                {"role": "tool", "tool_call_id": tc.id, "content": call["result"]}
                for tc, call in zip(msg.tool_calls, state["tool_calls_made"][-len(msg.tool_calls):])
            ]
            final = client.chat.completions.create(model=MODEL, messages=follow_up)
            state["answer"] = final.choices[0].message.content
        else:
            state["answer"] = msg.content
 
        state["history"].append({"role": "user", "content": state["query"]})
        state["history"].append({"role": "assistant", "content": state["answer"]})
    return state
 
 
def repo_analyst_node(state: DevPulseState) -> DevPulseState:
    persona = ("You are the Repo Analyst. Use search_repositories or get_repo_details "
               "or list_contributors to answer with real, current GitHub data.")
    return asyncio.run(_run_specialist(
        state, persona, {"search_repositories", "get_repo_details", "list_contributors"}
    ))
 
 
def issue_triage_node(state: DevPulseState) -> DevPulseState:
    persona = ("You are the Issue Triage Agent. Use list_open_issues to answer with "
               "real, currently-open issues. Briefly note priority where relevant.")
    return asyncio.run(_run_specialist(state, persona, {"list_open_issues"}))
 
 
def release_notes_node(state: DevPulseState) -> DevPulseState:
    persona = ("You are the Release Notes Agent. Use get_latest_release to answer "
               "with the real, current latest version and notes.")
    return asyncio.run(_run_specialist(state, persona, {"get_latest_release"}))
 
 
# -- BUILD THE GRAPH -----------------------------------------------------------
 
builder = StateGraph(DevPulseState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("repo_info", repo_analyst_node)
builder.add_node("issue_triage", issue_triage_node)
builder.add_node("release_notes", release_notes_node)
 
builder.add_conditional_edges(
    "classify_intent", route,
    {"repo_info": "repo_info", "issue_triage": "issue_triage", "release_notes": "release_notes"},
)
builder.set_entry_point("classify_intent")
builder.set_finish_point("repo_info")
builder.set_finish_point("issue_triage")
builder.set_finish_point("release_notes")
 
graph = builder.compile()
 
 
def main():
    if not check_connection(client, MODEL, BACKEND):
        return
    print(f"DevPulse (LangGraph multi-agent) ready (model: {MODEL}). Type 'quit' to exit.\n")
    history = []
    while True:
        query = input("You: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue
        result = graph.invoke({
            "query": query, "history": history, "category": "",
            "tool_calls_made": [], "answer": "",
        })
        history = result["history"]
        print(f"DevPulse [{result['category']}]: {result['answer']}\n")
 
 
if __name__ == "__main__":
    main()
 
 