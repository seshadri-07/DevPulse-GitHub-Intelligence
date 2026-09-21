"""DevPulse Studio — Streamlit demo UI for the MCP + LangGraph project.

Run with: streamlit run devpulse_app.py
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv

from mcp_bridge import mcp_session, discover_openai_tools, call_mcp_tool
from model_client import get_client_and_model


# ---------------------------------------------------------------------------
# Configuration and presentation
# ---------------------------------------------------------------------------

load_dotenv()
st.set_page_config(
    page_title="DevPulse intelligence chat bot",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "DevPulse "
MAX_TOOL_ROUNDS = 3
ROUTES = {
    "repo_info": {
        "title": "Repository analyst",
        "icon": "◫",
        "description": "Repository search, project health, and contributors",
        "tools": {"search_repositories", "get_repo_details", "list_contributors"},
    },
    "issue_triage": {
        "title": "Issue triage agent",
        "icon": "◉",
        "description": "Open issues and focused triage",
        "tools": {"list_open_issues"},
    },
    "release_notes": {
        "title": "Release analyst",
        "icon": "◇",
        "description": "Versions, releases, and release notes",
        "tools": {"get_latest_release"},
    },
}

SYSTEM_PROMPT = """You are DevPulse, a professional GitHub intelligence assistant.
Answer from live MCP tool results whenever a fact could be current. Be concise,
accurate, and readable. You may remember repositories mentioned earlier. Never
invent tool results. If an action needs data, call the relevant available tool."""

ROUTER_PROMPT = """Classify the user query into exactly one category:
- repo_info: repository search, stats, contributors, general repository questions
- issue_triage: open issues, bugs, issue status, triage
- release_notes: releases, versions, changelogs

Reply only with one category name. Query: {query}"""

CSS = """
<style>
    :root { color-scheme: dark; }
    .stApp { background: #0b1020; color: #e8edf8; }
    [data-testid="stSidebar"] { background: #10182d; border-right: 1px solid #263454; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
    .block-container { max-width: 1450px; padding-top: 1.2rem; padding-bottom: 1.5rem; }
    #MainMenu, footer { visibility: hidden; }
    .brand { display:flex; align-items:center; gap:11px; margin: 2px 0 22px; }
    .brand-mark { width:38px; height:38px; display:grid; place-items:center; border-radius:12px;
      background:linear-gradient(145deg,#7c5cff,#27c5d9); color:white; font-size:21px; box-shadow:0 6px 25px #6650cc66; }
    .brand h1 { font-size:1.38rem; margin:0; letter-spacing:-.035em; color:#f5f7ff; }
    .brand p { font-size:.74rem; margin:1px 0 0; color:#94a3c7; }
    .status { display:flex; align-items:center; gap:7px; color:#9fb0d8; font-size:.78rem; margin: 4px 0 18px; }
    .dot { width:8px; height:8px; background:#36d399; border-radius:50%; box-shadow:0 0 9px #36d399; }
    .hero { border:1px solid #263454; border-radius:18px; padding:1.15rem 1.3rem; margin-bottom:1rem;
      background:linear-gradient(115deg,#121b35 0%,#0e1730 56%,#15172e 100%); }
    .hero h2 { font-size:1.23rem; margin:0 0 .28rem; color:#f4f6fc; }
    .hero p { margin:0; color:#9eacce; font-size:.88rem; }
    .metric-card { min-height:80px; border-radius:13px; padding:12px 14px; background:#121b31; border:1px solid #263454; }
    .metric-card .label { font-size:.72rem; color:#91a1c8; text-transform:uppercase; letter-spacing:.06em; }
    .metric-card .value { font-size:1.3rem; font-weight:700; margin-top:6px; color:#f0f3fc; }
    .chat-empty { text-align:center; padding:5.6rem 1rem 4rem; color:#99a8c8; }
    .chat-empty .orb { width:62px; height:62px; margin:0 auto 16px; border-radius:20px; display:grid; place-items:center;
      background:linear-gradient(145deg,#7253e8,#22bbd0); font-size:30px; color:white; box-shadow:0 14px 35px #31277788; }
    .chat-empty h3 { color:#edf1fb; margin:0 0 7px; font-size:1.28rem; }
    .tool-card { border:1px solid #2b3a60; background:#111a31; border-radius:13px; padding:12px 14px; margin:7px 0; }
    .tool-head { display:flex; align-items:center; gap:8px; font-size:.82rem; font-weight:650; color:#dbe6ff; }
    .tool-icon { width:24px; height:24px; display:grid; place-items:center; border-radius:7px; background:#25375f; color:#79b9ff; }
    .tool-args { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.72rem; color:#aebde0; padding:7px 0 0 32px; overflow-wrap:anywhere; }
    .tool-result { border-left:2px solid #31c48d; padding:7px 0 0 10px; margin:9px 0 0 32px; font-size:.78rem; color:#c1cee6; white-space:pre-wrap; }
    .pending { border-color:#8e6f25; background:#1d1a27; }
    .log-line { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.71rem; padding:7px 1px; border-bottom:1px solid #24304d; color:#b4c2dd; }
    .log-time { color:#7385ae; margin-right:7px; }
    .graph { display:grid; gap:10px; padding:4px 0; }
    .node { border:1px solid #34456e; background:#131d36; border-radius:10px; padding:10px; font-size:.78rem; color:#cbd7f0; }
    .node.active { border-color:#6c56da; background:#211d48; box-shadow:0 0 0 1px #6c56da55; color:#fff; }
    .edge { width:1px; height:12px; background:#4d5f88; margin:-2px auto; }
    .pill { display:inline-block; margin-top:7px; padding:3px 7px; border-radius:99px; font-size:.66rem; background:#243656; color:#9cc7ff; }
    .stChatMessage { border-radius:14px; }
    [data-testid="stChatMessage"] { background:#121b31; border:1px solid #263454; margin-bottom:.65rem; }
    [data-testid="stChatInput"] { border-radius:14px; }
    .stButton button { border-radius:9px; font-weight:600; }
</style>
"""


@dataclass
class ToolEvent:
    """A serialisable record shown in the live execution area."""
    id: str
    name: str
    arguments: dict[str, Any]
    status: str = "pending"  # pending | running | approved | rejected | complete | failed
    result: str = ""
    created_at: str = ""
    tool_call_id: str = ""


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_async(coro: Any) -> Any:
    """Run a coroutine in Streamlit's synchronous execution model."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def initialise_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "agent_history": [],
        "tool_events": [],
        "logs": [],
        "conversation_archive": [],
        "conversation_id": str(uuid.uuid4()),
        "conversation_started_at": now(),
        "mode": "Single Agent",
        "backend": os.getenv("LLM_BACKEND", "ollama").lower(),
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        "pending_request": None,
        "active_route": None,
        "turn_count": 0,
        "last_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def conversation_title(messages: list[dict[str, Any]]) -> str:
    """Use the first user question as a compact, ChatGPT-style chat title."""
    first_question = next((item["content"] for item in messages if item["role"] == "user"), "New conversation")
    return first_question.replace("\n", " ").strip()[:48] or "New conversation"


def save_current_conversation() -> None:
    """Snapshot the current thread so it can be reopened from the sidebar.

    Streamlit session state is browser-session scoped; this archive intentionally
    stores no API keys and is not written to the local filesystem.
    """
    messages = st.session_state.messages
    if not messages:
        return
    record = {
        "id": st.session_state.conversation_id,
        "title": conversation_title(messages),
        "created_at": st.session_state.conversation_started_at,
        "updated_at": now(),
        "messages": copy.deepcopy(messages),
        "agent_history": copy.deepcopy(st.session_state.agent_history),
        "tool_events": [asdict(event) for event in st.session_state.tool_events],
        "mode": st.session_state.mode,
        "active_route": st.session_state.active_route,
    }
    archive = st.session_state.conversation_archive
    index = next((i for i, item in enumerate(archive) if item["id"] == record["id"]), None)
    if index is None:
        archive.insert(0, record)
    else:
        archive[index] = record
    st.session_state.conversation_archive = archive[:20]


def start_new_conversation() -> None:
    save_current_conversation()
    st.session_state.messages = []
    st.session_state.agent_history = []
    st.session_state.tool_events = []
    st.session_state.logs = []
    st.session_state.pending_request = None
    st.session_state.active_route = None
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.conversation_started_at = now()
    log("New conversation created")


def open_conversation(conversation_id: str) -> None:
    """Restore a saved conversation, including its visible tool cards."""
    if conversation_id == st.session_state.conversation_id:
        return
    save_current_conversation()
    record = next(item for item in st.session_state.conversation_archive if item["id"] == conversation_id)
    st.session_state.conversation_id = record["id"]
    st.session_state.conversation_started_at = record["created_at"]
    st.session_state.messages = copy.deepcopy(record["messages"])
    st.session_state.agent_history = copy.deepcopy(record["agent_history"])
    st.session_state.tool_events = [ToolEvent(**event) for event in record["tool_events"]]
    st.session_state.mode = record.get("mode", "Single Agent")
    st.session_state.active_route = record.get("active_route")
    st.session_state.pending_request = None
    st.session_state.logs = []
    log(f"Opened saved conversation: {record['title']}")


def log(message: str, level: str = "INFO") -> None:
    st.session_state.logs.insert(0, {"time": now(), "level": level, "message": message})
    st.session_state.logs = st.session_state.logs[:80]


def configure_backend(backend: str, model: str, endpoint: str, groq_key: str) -> None:
    """Apply sidebar settings only to this Streamlit process."""
    os.environ["LLM_BACKEND"] = backend
    if backend == "ollama":
        os.environ["OLLAMA_MODEL"] = model
        os.environ["OLLAMA_BASE_URL"] = endpoint.rstrip("/") + "/v1"
    else:
        os.environ["GROQ_MODEL"] = model
        if groq_key.strip():
            os.environ["GROQ_API_KEY"] = groq_key.strip()


def get_llm() -> tuple[Any, str, str]:
    return get_client_and_model()


def response_content(response: Any) -> str:
    return response.choices[0].message.content or ""


def normalise_tool_calls(message: Any) -> list[ToolEvent]:
    calls = getattr(message, "tool_calls", None) or []
    events = []
    for call in calls:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"raw_arguments": call.function.arguments}
        events.append(ToolEvent(
            id=str(uuid.uuid4()), name=call.function.name, arguments=arguments,
            created_at=now(), tool_call_id=call.id,
        ))
    return events


def client_call(messages: list[dict], tools: Optional[list[dict]] = None) -> Any:
    client, model, backend = get_llm()
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        kwargs.update({"tools": tools, "tool_choice": "auto"})
    log(f"Calling {backend.title()} model: {model}")
    return client.chat.completions.create(**kwargs)


async def discover_tools() -> list[dict]:
    async with mcp_session() as session:
        return await discover_openai_tools(session)


async def execute_tools(events: list[ToolEvent]) -> list[ToolEvent]:
    """Use one live MCP session for all approved calls in a turn."""
    async with mcp_session() as session:
        for event in events:
            if event.status != "approved":
                continue
            event.status = "running"
            log(f"MCP → {event.name}({json.dumps(event.arguments)})")
            try:
                event.result = await call_mcp_tool(session, event.name, event.arguments)
                event.status = "complete"
                log(f"MCP ← {event.name} completed")
            except Exception as exc:  # status must stay visible in the demo UI
                event.status = "failed"
                event.result = f"Tool execution failed: {exc}"
                log(f"MCP tool failed: {event.name} — {exc}", "ERROR")
    return events


def classify_route(query: str) -> str:
    """Use the model for routing, with keyword fallback for offline demos."""
    try:
        reply = response_content(client_call([
            {"role": "system", "content": "You are a precise classifier."},
            {"role": "user", "content": ROUTER_PROMPT.format(query=query)},
        ])).strip().lower()
        for route in ROUTES:
            if route in reply:
                log(f"LangGraph router selected: {route}")
                return route
    except Exception as exc:
        log(f"Router model call unavailable; applying local route fallback ({exc})", "WARN")
    lowered = query.lower()
    if any(word in lowered for word in ("issue", "bug", "triage", "ticket")):
        return "issue_triage"
    if any(word in lowered for word in ("release", "version", "changelog")):
        return "release_notes"
    return "repo_info"


def allowed_tools(all_tools: list[dict], route: Optional[str]) -> list[dict]:
    if not route:
        return all_tools
    permitted = ROUTES[route]["tools"]
    return [tool for tool in all_tools if tool["function"]["name"] in permitted]


def assistant_message_with_calls(message: Any) -> dict[str, Any]:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        calls.append(call.model_dump())
    return {"role": "assistant", "content": message.content or "", "tool_calls": calls}


def launch_turn(query: str) -> None:
    """Start a request and pause if the LLM proposes an MCP action."""
    st.session_state.turn_count += 1
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.agent_history.append({"role": "user", "content": query})
    save_current_conversation()
    st.session_state.last_error = ""
    route = classify_route(query) if st.session_state.mode == "LangGraph Multi-Agent" else None
    st.session_state.active_route = route
    try:
        all_tools = run_async(discover_tools())
        scope = allowed_tools(all_tools, route)
        if route:
            config = ROUTES[route]
            system = f"{SYSTEM_PROMPT}\nYou are the {config['title']}. Restrict yourself to your assigned tools."
        else:
            system = SYSTEM_PROMPT
        messages = [{"role": "system", "content": system}] + st.session_state.agent_history
        answer = client_call(messages, scope)
        message = answer.choices[0].message
        events = normalise_tool_calls(message)
        if events:
            st.session_state.agent_history.append(assistant_message_with_calls(message))
            st.session_state.tool_events.extend(events)
            st.session_state.pending_request = {
                "events": [event.id for event in events], "route": route, "round": 1,
            }
            log(f"Approval requested for {len(events)} MCP tool call(s)")
            save_current_conversation()
        else:
            final = message.content or "I couldn't generate a response."
            finish_turn(final)
    except Exception as exc:
        st.session_state.last_error = str(exc)
        log(f"Request failed: {exc}", "ERROR")
        finish_turn("I couldn't reach the configured model or MCP service. Check the connection details in the sidebar.")


def finish_turn(answer: str) -> None:
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.agent_history.append({"role": "assistant", "content": answer})
    st.session_state.pending_request = None
    log("Assistant response completed")
    save_current_conversation()


def find_event(event_id: str) -> ToolEvent:
    for item in st.session_state.tool_events:
        if item.id == event_id:
            return item
    raise KeyError(event_id)


def continue_after_review() -> None:
    """Execute approved tools, then let the same agent synthesise a final answer."""
    pending = st.session_state.pending_request
    if not pending:
        return
    events = [find_event(event_id) for event_id in pending["events"]]
    if not any(event.status == "approved" for event in events):
        for event in events:
            if event.status == "pending":
                event.status = "rejected"
        for event in events:
            st.session_state.agent_history.append({
                "role": "tool", "tool_call_id": event.tool_call_id,
                "content": "Tool execution denied by the human reviewer.",
            })
        finish_turn("No live tool was executed because the requested action was declined.")
        return

    try:
        run_async(execute_tools(events))
        for event in events:
            result = event.result if event.status == "complete" else "Tool execution denied by human reviewer."
            st.session_state.agent_history.append({"role": "tool", "tool_call_id": event.tool_call_id, "content": result})
        route = pending.get("route")
        system = SYSTEM_PROMPT
        if route:
            system += f"\nYou are the {ROUTES[route]['title']}."
        response = client_call([{"role": "system", "content": system}] + st.session_state.agent_history)
        finish_turn(response_content(response) or "The live results are shown in the tool card above.")
    except Exception as exc:
        log(f"Post-tool synthesis failed: {exc}", "ERROR")
        finish_turn("The tool ran, but the model could not produce the final summary. See the tool result above.")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_brand() -> None:
    st.markdown("""<div class='brand'><div class='brand-mark'>◈</div><div><h1>DevPulse</h1>
    <p>MCP GitHub intelligence</p></div></div>""", unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        render_brand()
        st.markdown("<div class='status'><span class='dot'></span>Studio session active</div>", unsafe_allow_html=True)
        st.caption("AGENT ARCHITECTURE")
        mode = st.radio("Run mode", ["Single Agent", "LangGraph Multi-Agent"], label_visibility="collapsed",
                        index=0 if st.session_state.mode == "Single Agent" else 1)
        st.session_state.mode = mode
        st.divider()
        st.caption("MODEL BACKEND")
        backend = st.selectbox("Provider", ["ollama", "groq"], index=0 if st.session_state.backend == "ollama" else 1)
        default_model = st.session_state.model
        if backend == "ollama":
            model = st.text_input("Ollama model", value=default_model if st.session_state.backend == "ollama" else "qwen2.5:3b")
            endpoint = st.text_input("Ollama endpoint", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
            groq_key = ""
        else:
            model = st.text_input("Groq model", value=default_model if st.session_state.backend == "groq" else "llama-3.1-8b-instant")
            groq_key = st.text_input("Groq API key", type="password", help="Kept only in this running process.")
            endpoint = ""
        if st.button("Apply model settings", use_container_width=True):
            configure_backend(backend, model, endpoint, groq_key)
            st.session_state.backend, st.session_state.model = backend, model
            log(f"Backend changed to {backend} / {model}")
            st.success("Settings applied")
        st.divider()
        st.caption("SESSION")
        if st.button("New conversation", use_container_width=True, disabled=bool(st.session_state.pending_request)):
            start_new_conversation()
            st.rerun()
        if st.session_state.conversation_archive:
            st.caption("RECENT CONVERSATIONS")
            if st.session_state.pending_request:
                st.caption("Finish the approval step before switching chats.")
            for record in st.session_state.conversation_archive:
                is_current = record["id"] == st.session_state.conversation_id
                label = f"● {record['title']}" if is_current else record["title"]
                if st.button(label, key=f"open_chat_{record['id']}", use_container_width=True,
                             disabled=is_current or bool(st.session_state.pending_request)):
                    open_conversation(record["id"])
                    st.rerun()
        st.caption("Tool calls always require your approval before live execution.")


def render_metrics() -> None:
    left, middle, right = st.columns(3)
    completed = sum(item.status == "complete" for item in st.session_state.tool_events)
    active = st.session_state.active_route
    with left:
        st.markdown(f"<div class='metric-card'><div class='label'>Mode</div><div class='value'>{'Multi-agent' if st.session_state.mode.startswith('Lang') else 'Single agent'}</div></div>", unsafe_allow_html=True)
    with middle:
        st.markdown(f"<div class='metric-card'><div class='label'>Live tools executed</div><div class='value'>{completed}</div></div>", unsafe_allow_html=True)
    with right:
        title = ROUTES[active]['title'] if active else "General agent"
        st.markdown(f"<div class='metric-card'><div class='label'>Active specialist</div><div class='value'>{title}</div></div>", unsafe_allow_html=True)


def render_tool_card(event: ToolEvent) -> None:
    css_class = "tool-card pending" if event.status == "pending" else "tool-card"
    status = event.status.replace("_", " ").title()
    safe_name = event.name.replace("_", " ")
    args = json.dumps(event.arguments, indent=2)
    st.markdown(f"""<div class='{css_class}'><div class='tool-head'><span class='tool-icon'>⌘</span>
    MCP Tool · {safe_name} <span class='pill'>{status}</span></div>
    <div class='tool-args'>{args}</div>""", unsafe_allow_html=True)
    if event.result:
        st.markdown(f"<div class='tool-result'>{event.result}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_conversation() -> None:
    if not st.session_state.messages:
        st.markdown("""<div class='chat-empty'><div class='orb'>◈</div><h3>GitHub intelligence, with a human in control.</h3>
        <p>Ask about repositories, open issues, contributors, or latest releases.</p></div>""", unsafe_allow_html=True)
        examples = st.columns(3)
        prompts = [
            "Compare the current health of facebook/react",
            "What are the top open issues in langchain-ai/langchain?",
            "Show the latest release for microsoft/vscode",
        ]
        for column, prompt in zip(examples, prompts):
            with column:
                if st.button(prompt, use_container_width=True, key=f"example_{prompt}"):
                    launch_turn(prompt)
                    st.rerun()
        return
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if st.session_state.tool_events:
        with st.expander("Live tool activity", expanded=bool(st.session_state.pending_request)):
            for event in st.session_state.tool_events:
                render_tool_card(event)


def render_approval() -> None:
    pending = st.session_state.pending_request
    if not pending:
        return
    events = [find_event(event_id) for event_id in pending["events"]]
    st.warning("Human approval required — review each live GitHub API action before it runs.")
    for event in events:
        st.code(f"{event.name}({json.dumps(event.arguments)})", language="json")

    # One action must be enough: approving this button immediately executes
    # every proposed tool and then asks the model to write its final answer.
    approve, decline = st.columns(2)
    with approve:
        if st.button("Approve & run live tool(s)", type="primary", use_container_width=True):
            for event in events:
                event.status = "approved"
                log(f"Human approved {event.name}")
            continue_after_review()
            st.rerun()
    with decline:
        if st.button("Decline tool request", use_container_width=True):
            for event in events:
                event.status = "rejected"
                log(f"Human declined {event.name}", "WARN")
            continue_after_review()
            st.rerun()


def render_graph() -> None:
    active = st.session_state.active_route
    st.subheader("LangGraph execution")
    if st.session_state.mode != "LangGraph Multi-Agent":
        st.info("Switch to Multi-Agent mode to view the router and specialist path.")
        return
    nodes = [("User request", True), ("Intent router", bool(active))]
    nodes += [(route["title"], key == active) for key, route in ROUTES.items()]
    for index, (label, is_active) in enumerate(nodes):
        klass = "node active" if is_active else "node"
        st.markdown(f"<div class='{klass}'>{label}</div>", unsafe_allow_html=True)
        if index < len(nodes) - 1:
            st.markdown("<div class='edge'></div>", unsafe_allow_html=True)
    if active:
        st.caption(f"Selected route: {ROUTES[active]['description']}")


def render_logs() -> None:
    st.subheader("MCP logs")
    if not st.session_state.logs:
        st.caption("Waiting for activity…")
        return
    for record in st.session_state.logs[:16]:
        colour = "#ff9b9b" if record["level"] == "ERROR" else "#ffcf70" if record["level"] == "WARN" else "#8ca4d3"
        st.markdown(f"<div class='log-line'><span class='log-time'>{record['time']}</span><span style='color:{colour}'>[{record['level']}]</span> {record['message']}</div>", unsafe_allow_html=True)


def main() -> None:
    initialise_state()
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()
    st.markdown(f"<div class='hero'><h2>{APP_NAME}</h2><p>Live GitHub research through MCP, model reasoning, and explicit human approval.</p></div>", unsafe_allow_html=True)
    render_metrics()
    st.write("")
    chat, right = st.columns([2.1, 1], gap="large")
    with chat:
        render_conversation()
        render_approval()
        prompt = st.chat_input("Ask DevPulse about GitHub…", disabled=bool(st.session_state.pending_request))
        if prompt:
            launch_turn(prompt)
            st.rerun()
    with right:
        render_graph()
        st.divider()
        render_logs()
    if st.session_state.last_error:
        with st.expander("Connection diagnostics"):
            st.code(st.session_state.last_error)


if __name__ == "__main__":
    main()
