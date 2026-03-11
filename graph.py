import os
import json
import uuid
from typing import Annotated, TypedDict, List, Dict, Any
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from tools.implementation import (
    getProfile, listProjects, explainProject, getAvailability, analyzeJobFit
)
from calendar_mcp import get_calendar_tools

load_dotenv()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    trace: Dict[str, Any]
    active_agent: str  # "portfolio" | "meet"

# ---------------------------------------------------------------------------
# LLM — llama-3.3-70b-versatile for tool calling; llama-3.1-8b-instant for cheap routing
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# Fast, low-cost model used only for the intent classifier (no tool calling needed)
llm_router = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# ---------------------------------------------------------------------------
# Portfolio tools (pure Python — no MCP, no async)
# ---------------------------------------------------------------------------
from langchain_core.tools import tool

@tool
def get_profile_tool() -> str:
    """Retrieves Mudasir's basic profile information, bio, and general experience."""
    return json.dumps(getProfile())

@tool
def list_projects_tool(filters: str = "") -> str:
    """Lists portfolio projects. Optionally pass comma-separated tech tags to filter, e.g. 'React,Python'."""
    filter_list = [f.strip() for f in filters.split(",")] if filters.strip() else None
    return json.dumps(listProjects(filter_list))

@tool
def explain_project_tool(name: str) -> str:
    """Gets in-depth technical details for a specific project by its exact name."""
    return json.dumps(explainProject(name))

@tool
def get_availability_tool() -> str:
    """Gets the developer's current employment status and availability for new roles."""
    return json.dumps(getAvailability())

@tool
def analyze_job_fit_tool(job_text: str) -> str:
    """Analyzes a job description and returns how well it matches Mudasir's skills."""
    return json.dumps(analyzeJobFit(job_text))

portfolio_tools = [
    get_profile_tool,
    list_projects_tool,
    explain_project_tool,
    get_availability_tool,
    analyze_job_fit_tool,
]

# ---------------------------------------------------------------------------
# Cached MCP tools — populated once when the graph is created, then reused.
# Keeping the same instances ensures the ToolNode and bind_tools() share one session.
# ---------------------------------------------------------------------------
_cal_read_tools: List = []   # list-calendars, list-events
_meet_tools: List = []       # create_meeting (etc.)

# ---------------------------------------------------------------------------
# Book-meeting proxy tool — uses a subprocess with its own event loop to avoid
# asyncio conflicts inside a synchronous LangGraph ToolNode.
# Payload is base64-encoded to prevent any string-escaping issues.
# ---------------------------------------------------------------------------
@tool
def book_meeting_tool(name: str, email: str, start_time: str, end_time: str, description: str = "") -> str:
    """
    Books a meeting on Mudasir's Google Calendar, auto-generates a Google Meet link,
    and sends an invitation email to the attendee.
    Args:
        name: Attendee's full name.
        email: Attendee's exact email address.
        start_time: ISO 8601 start, e.g. '2026-03-15T14:00:00-05:00'. Include timezone offset.
        end_time: ISO 8601 end, e.g. '2026-03-15T15:00:00-05:00'. Include timezone offset.
        description: Optional short context for the meeting body.
    """
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    def add_tz(ts: str) -> str:
        """Default to PKT (UTC+5) if no offset is present."""
        if "+" not in ts and ts.upper()[-1] != "Z" and "-" not in ts[10:]:
            return ts + "+05:00"
        return ts

    script_dir = os.path.dirname(os.path.abspath(__file__))
    creds_path = os.path.join(script_dir, "credentials.json")
    # GOOGLE_TOKEN_PATH env var lets you point to the token on Railway's persistent disk
    token_path = os.getenv(
        "GOOGLE_TOKEN_PATH",
        os.path.join(os.path.expanduser("~"), ".config", "google-calendar-mcp", "tokens.json"),
    )

    if not os.path.exists(creds_path):
        return json.dumps({"status": "error", "message": "credentials.json not found."})
    if not os.path.exists(token_path):
        return json.dumps({"status": "error", "message": "OAuth token not found. Please re-authenticate."})

    # Load tokens — the google-calendar-mcp stores { "normal": { access_token, refresh_token, ... } }
    with open(token_path) as f:
        token_data = json.load(f)

    # Support both { "normal": {...} } (dict of accounts) and legacy list formats
    if isinstance(token_data, dict) and "normal" in token_data:
        raw_token = token_data["normal"]
    elif isinstance(token_data, list):
        for entry in token_data:
            if entry.get("account") == "normal":
                raw_token = entry.get("token") or entry
                break
        else:
            raw_token = token_data[0].get("token") or token_data[0]
    else:
        raw_token = token_data

    with open(creds_path) as f:
        cred_info = json.load(f)
    web = cred_info.get("web") or cred_info.get("installed", {})

    # Pass expiry so Credentials.valid / .expired work correctly
    expiry_date_ms = raw_token.get("expiry_date")
    token_expiry = (
        datetime.utcfromtimestamp(expiry_date_ms / 1000) if expiry_date_ms else None
    )

    creds = Credentials(
        token=raw_token.get("access_token"),
        refresh_token=raw_token.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=web.get("client_id"),
        client_secret=web.get("client_secret"),
        expiry=token_expiry,
    )

    # Auto-refresh if expired and persist the new token so future calls don't need to refresh
    if not creds.valid:
        try:
            from google.auth.transport.requests import Request as GRequest
            creds.refresh(GRequest())
            if isinstance(token_data, dict) and "normal" in token_data:
                token_data["normal"]["access_token"] = creds.token
                if creds.expiry:
                    _epoch = datetime(1970, 1, 1)
                    token_data["normal"]["expiry_date"] = int(
                        (creds.expiry - _epoch).total_seconds() * 1000
                    )
                with open(token_path, "w") as _tf:
                    json.dump(token_data, _tf, indent=2)
        except Exception as _re:
            return json.dumps({"status": "error", "message": f"Session expired — please re-authenticate. ({_re})"})

    try:
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)

        # Working-hours guard: 9 AM – 6 PM PKT
        try:
            _pkt = timezone(timedelta(hours=5))
            _dt_start = datetime.fromisoformat(add_tz(start_time))
            _hour_pkt = _dt_start.astimezone(_pkt).hour
            if _hour_pkt < 9 or _hour_pkt >= 18:
                _fmt = _dt_start.astimezone(_pkt).strftime("%I:%M %p PKT")
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"The requested time ({_fmt}) is outside Mudasir's available hours "
                        f"(9:00 AM \u2013 6:00 PM PKT). Please suggest a time within working hours."
                    ),
                })
        except (ValueError, TypeError):
            pass  # unparseable timestamp — let Calendar API validate

        # Conflict check — skip silently if the calendar query itself fails
        try:
            _existing = service.events().list(
                calendarId="primary",
                timeMin=add_tz(start_time),
                timeMax=add_tz(end_time),
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            _conflicts = [
                e for e in _existing.get("items", []) if e.get("status") != "cancelled"
            ]
            if _conflicts:
                _titles = ", ".join(
                    e.get("summary", "an existing event") for e in _conflicts[:2]
                )
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"Mudasir already has a booking at that time ({_titles}). "
                        f"Could you suggest a different time?"
                    ),
                })
        except HttpError:
            pass  # proceed — let insert handle any conflict

        event = {
            "summary": f"Meeting with {name}",
            "description": description.strip() if description else "",
            "start": {"dateTime": add_tz(start_time), "timeZone": "UTC"},
            "end": {"dateTime": add_tz(end_time), "timeZone": "UTC"},
            "attendees": [{"email": email}],
            "conferenceData": {
                "createRequest": {
                    "requestId": str(uuid.uuid4()),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
            "reminders": {"useDefault": True},
        }

        created = service.events().insert(
            calendarId="primary",
            body=event,
            conferenceDataVersion=1,
            sendUpdates="all",
        ).execute()

        meet_link = (
            created.get("hangoutLink")
            or created.get("conferenceData", {})
            .get("entryPoints", [{}])[0]
            .get("uri", "")
        )

        return json.dumps({
            "status": "success",
            "event_id": created.get("id"),
            "event_link": created.get("htmlLink"),
            "meet_link": meet_link,
            "message": f"Meeting booked for {name} ({email}) — {start_time}",
        })

    except HttpError as e:
        return json.dumps({"status": "error", "message": f"Google Calendar API error: {e}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

# Keywords that indicate the user wants to cancel / exit the booking flow
_CANCEL_KEYWORDS = {"no", "nope", "nevermind", "never mind", "stop", "cancel", "forget it"}

# Words that count as an affirmative reply (also covers voice STT variants and common non-English)
_AFFIRMATIVE = {
    "yes", "sure", "ok", "okay", "yeah", "yep", "please", "alright",
    "go ahead", "sounds good", "let's", "lets", "absolutely", "definitely",
    # Non-English affirmatives (user may speak French, Urdu, etc.)
    "oui", "ja", "si", "sí", "da", "haan", "bilkul", "zaroor", "ji", "hai",
}

# Phrases the portfolio agent uses when offering to book a meeting
_MEETING_OFFER_PHRASES = (
    "schedule a meeting", "book a meeting", "schedule a call", "set up a meeting",
    "arrange a meeting", "talk to mudasir", "connect with mudasir", "discuss directly",
    "schedule time", "book a time", "would you like to meet", "should i schedule",
    "want me to schedule", "want to schedule",
)

def identify_intent(state: AgentState) -> dict:
    """
    Routes to 'portfolio' or 'meet'.

    Priority order:
    1. Cancel keywords       → always 'portfolio'
    2. Already in 'meet'     → stay 'meet' (sticky)
    3. Affirmative reply to a meeting offer from the portfolio agent → 'meet'
    4. LLM classification
    """
    messages = state["messages"]
    last_human = messages[-1].content.strip().lower()
    current_agent = state.get("active_agent", "portfolio")

    # Priority 1: explicit cancellation
    if any(kw in last_human for kw in _CANCEL_KEYWORDS):
        return {"active_agent": "portfolio"}

    # Priority 2: already in the meet flow — stay sticky
    if current_agent == "meet":
        return {"active_agent": "meet"}

    # Priority 3: user replied affirmatively to a meeting offer from the portfolio agent
    # Strip punctuation so STT output like "Yes." or "Sure!" still matches
    import re as _re
    last_human_clean = _re.sub(r"[^\w\s]", "", last_human).strip()
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage):
            ai_lower = msg.content.lower()
            if any(phrase in ai_lower for phrase in _MEETING_OFFER_PHRASES):
                words = set(last_human_clean.split())
                if words & _AFFIRMATIVE or last_human_clean in _AFFIRMATIVE:
                    return {"active_agent": "meet"}
            break  # only inspect the most recent AI turn — must be inside isinstance check

    # Priority 4: LLM classification (uses cheap small model — no tools needed)
    system_prompt = (
        "Classify the user's message as 'portfolio' or 'meet'. Reply with exactly one lowercase word.\n"
        "Choose 'meet' if the user mentions scheduling, booking, meeting, appointment, calendar, "
        "a specific time, or a date.\n"
        "Choose 'portfolio' for everything else."
    )
    response = llm_router.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=messages[-1].content),
    ])
    decision = response.content.lower().strip()
    return {"active_agent": "meet" if "meet" in decision or "calendar" in decision else "portfolio"}


async def portfolio_chatbot(state: AgentState) -> dict:
    """Handles portfolio Q&A using pure-Python tools."""
    from langchain_core.messages import ToolMessage

    llm_with_tools = llm.bind_tools(portfolio_tools)
    portfolio_tool_names = {t.name for t in portfolio_tools}

    # Defensive: remove any AI messages that contain non-portfolio tool calls
    # (e.g. book_meeting_tool calls that bled in from a previous meet session).
    # Also remove orphaned ToolMessages whose call_id has no matching AI message.
    ids_to_skip: set = set()
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if any(tc["name"] not in portfolio_tool_names for tc in msg.tool_calls):
                ids_to_skip.update(tc["id"] for tc in msg.tool_calls)

    safe_messages = [
        msg for msg in state["messages"]
        if not (
            (isinstance(msg, AIMessage) and msg.tool_calls and
             any(tc["name"] not in portfolio_tool_names for tc in msg.tool_calls))
            or
            (isinstance(msg, ToolMessage) and msg.tool_call_id in ids_to_skip)
        )
    ]

    system_prompt = (
        "You are 'Noir AI', Mudasir Shah's personal AI concierge.\n\n"
        "YOUR ONLY JOB is to showcase Mudasir's background and guide interested visitors toward booking "
        "a meeting with him. You are NOT a general-purpose assistant, search engine, or consultant.\n\n"

        "RULES:\n"
        "1. TOOLS FIRST: Always call a tool to fetch facts before answering. Never answer from memory.\n"
        "2. CONCISE: Give short, direct answers. Expand only when explicitly asked.\n\n"

        "3. OFF-TOPIC REDIRECT — This is the most important rule.\n"
        "   If the user asks about ANYTHING not directly related to Mudasir Shah's work, skills, "
        "projects, availability, or booking a meeting, do NOT answer it. "
        "Always respond with a polite one-liner that makes your purpose clear, then offer what you CAN do. "
        "Use this exact pattern every time:\n"
        "   'I'm only here to help with Mudasir Shah's portfolio and to schedule meetings with him. "
        "I can't help with [brief restatement of their topic]. "
        "What I *can* do is tell you about his work, skills, or projects — or book a meeting with him. "
        "Which would you like?'\n"
        "   Examples of off-topic: general knowledge, news, weather, coding tutorials, opinions, "
        "recipes, math, other people, or anything unrelated to Mudasir.\n\n"

        "4. PROJECT / TECHNICAL WORK REQUESTS: If the user asks you to design, plan, estimate, or "
        "build something, do NOT engage with the specifics. Say: "
        "'That sounds like a great project \u2014 Mudasir would be the right person to discuss the details "
        "with. Would you like me to schedule a meeting with him right now?'\n\n"

        "5. MEETING HANDOFF: Whenever a user shows genuine interest (wants to hire him, work with him, "
        "discuss a project, or asks to talk to him), proactively offer to book a meeting. "
        "Use this exact phrasing so routing works: "
        "'Would you like me to schedule a meeting with Mudasir?'\n\n"

        "6. NEVER output raw function call syntax like <function=tool_name> or JSON blobs. "
        "Only return clean, human-readable prose.\n\n"

        "7. LANGUAGE: Always reply in English.\n"
        "8. TONE: Confident, warm, and professional."
    )

    # Trim history to keep token usage low — system prompt + last 12 exchanges
    messages = [SystemMessage(content=system_prompt)] + safe_messages[-12:]
    response = await llm_with_tools.ainvoke(messages)

    trace = state.get("trace", {})
    trace["active_agent"] = "Portfolio Agent"
    if response.tool_calls:
        trace["intent_detected"] = "Tool Execution"
        trace["tool_selected"] = response.tool_calls[0]["name"]
        trace["tool_args"] = response.tool_calls[0]["args"]
    else:
        trace.setdefault("intent_detected", "Direct Answer")
        trace.setdefault("tool_selected", "None")
        trace.setdefault("tool_args", {})

    return {"messages": [response], "trace": trace}


async def meet_chatbot(state: AgentState) -> dict:
    """Handles calendar and meeting scheduling using cached MCP tools."""
    trace = state.get("trace", {})
    trace["active_agent"] = "Meet Agent"

    # Use module-level cached tools — same instances as in the meet_tools ToolNode
    meet_agent_tools = _cal_read_tools + [book_meeting_tool]

    if not meet_agent_tools:
        return {
            "messages": [AIMessage(content=(
                "I'm sorry, I can't access the calendar right now. "
                "Please ensure credentials are configured and try again."
            ))],
            "trace": trace,
        }

    llm_with_tools = llm.bind_tools(meet_agent_tools)
    current_time = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

    system_prompt = (
        f"You are 'Noir AI', Mudasir Shah's personal concierge and AI assistant.\n"
        f"Today is {current_time} PKT (Pakistan Standard Time, UTC+5).\n\n"

        "=== COLLECTING DETAILS ===\n"
        "You need exactly 3 things to book a meeting. Ask for any that are MISSING — one per turn:\n"
        "  1. Attendee's full name\n"
        "  2. Attendee's email address\n"
        "  3. Preferred date and time\n\n"
        "STRICT RULES for collection:\n"
        "  - NEVER invent, assume, or guess any missing detail. If it hasn't been explicitly stated, ask for it.\n"
        "  - NEVER discard a detail the user has already given — carry name and email across every turn.\n"
        "  - You MAY ask about the meeting topic or purpose — this helps Mudasir prepare. But ask it only ONCE and only after you have the name.\n"
        "  - Only proceed to booking once ALL 3 required details are confirmed in this conversation.\n"
        "  - If the user provides all 3 in one message, book immediately — do not ask again.\n\n"
        "Timezone:\n"
        "  - ALWAYS assume PKT (UTC+5) unless the user explicitly states otherwise.\n"
        "  - NEVER ask the user about timezones or date formats — just accept natural language like '18 march 9pm'.\n"
        "  - Use the EXACT time the user states — do NOT round, adjust, or shift. 'around 3 pm' means 3:00 PM.\n"
        "  - Email via voice: if the user says 'at the rate of' or 'at sign', treat it as '@'.\n\n"

        "=== BOOKING ===\n"
        "When ALL 3 details are confirmed, call book_meeting_tool ONCE — do NOT ask for confirmation first.\n"
        "  - Convert the time to ISO 8601 with PKT offset: e.g. 2026-03-18T21:00:00+05:00\n"
        "  - Default duration: 1 hour.\n"
        "  - Set description to the meeting topic/purpose if the user mentioned one; otherwise leave blank.\n\n"

        "=== ON SUCCESS ===\n"
        "Reply warmly and concisely with:\n"
        "  - The date and time in plain English (e.g. 'March 18 at 9:00 PM PKT')\n"
        "  - The Google Meet link from meet_link\n"
        "  - A note that a calendar invite has been sent to their email\n\n"

        "=== ON ERROR ===\n"
        "Working hours error (9 AM–6 PM PKT): Do NOT apologize repeatedly. Say once that the time is outside working hours, "
        "suggest a concrete alternative time (e.g. '3:00 PM PKT'), and ask the user to confirm or pick a different time.\n"
        "Other errors: Keep name/email already collected. Ask only for missing or corrected info, then retry immediately.\n\n"

        "=== FINAL RULES ===\n"
        "- Never claim the meeting is booked until the tool returns status='success'.\n"
        "- Never expose ISO timestamps, UTC offsets, or any technical jargon to the user.\n"
        "- Never output raw function call syntax.\n"
        "- Always reply in English."
    )

    # Trim history to keep token usage low — system prompt + last 12 exchanges
    messages = [SystemMessage(content=system_prompt)] + state["messages"][-12:]
    response = await llm_with_tools.ainvoke(messages)

    if response.tool_calls:
        trace["intent_detected"] = "Tool Execution"
        trace["tool_selected"] = response.tool_calls[0]["name"]
        trace["tool_args"] = response.tool_calls[0]["args"]
    else:
        trace.setdefault("intent_detected", "Direct Answer")
        trace.setdefault("tool_selected", "None")
        trace.setdefault("tool_args", {})

    return {"messages": [response], "trace": trace}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_decision(state: AgentState) -> str:
    return "meet_chatbot" if state["active_agent"] == "meet" else "portfolio_chatbot"


def portfolio_should_continue(state: AgentState) -> str:
    return "portfolio_tools" if state["messages"][-1].tool_calls else END


def meet_should_continue(state: AgentState) -> str:
    return "meet_tools" if state["messages"][-1].tool_calls else END


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

async def create_portfolio_graph():
    global _cal_read_tools, _meet_tools

    # Load read-only calendar MCP tools (list calendars/events) — used by meet_chatbot for context
    cal_tools = await get_calendar_tools()
    _cal_read_tools = [t for t in cal_tools if t.name in {"list-calendars", "list-events"}]
    _meet_tools = []  # Google Meet MCP replaced by built-in conferencing in book_meeting_tool

    # Separate ToolNodes: each agent only invokes its own tool set
    portfolio_tool_node = ToolNode(portfolio_tools)
    meet_tool_node = ToolNode(_cal_read_tools + [book_meeting_tool])

    workflow = StateGraph(AgentState)
    workflow.add_node("identify_intent", identify_intent)
    workflow.add_node("portfolio_chatbot", portfolio_chatbot)
    workflow.add_node("meet_chatbot", meet_chatbot)
    workflow.add_node("portfolio_tools", portfolio_tool_node)
    workflow.add_node("meet_tools", meet_tool_node)

    workflow.add_edge(START, "identify_intent")
    workflow.add_conditional_edges(
        "identify_intent", route_decision, ["portfolio_chatbot", "meet_chatbot"]
    )
    workflow.add_conditional_edges(
        "portfolio_chatbot", portfolio_should_continue, ["portfolio_tools", END]
    )
    workflow.add_conditional_edges(
        "meet_chatbot", meet_should_continue, ["meet_tools", END]
    )
    # After tools execute, return to the owning agent (unconditional)
    workflow.add_edge("portfolio_tools", "portfolio_chatbot")
    workflow.add_edge("meet_tools", "meet_chatbot")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
