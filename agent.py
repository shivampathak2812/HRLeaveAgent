import hashlib
import os
import threading
import pandas as pd

from dotenv import load_dotenv
from pypdf import PdfReader
from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import faq


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file")


# ============================================================
# 2. FILE PATHS
# ============================================================

# Paths are relative to this file, so the agent works from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sample files that ship with the project
DEFAULT_RULE_BOOK_PATH = os.path.join(BASE_DIR, "DIY - HR - Rule Book.pdf")
DEFAULT_EMPLOYEE_DB_PATH = os.path.join(BASE_DIR, "DIY - HR - Employee DB.xlsx")

# Columns the employee database must have
REQUIRED_COLUMNS = [
    "EmployeeID", "Name", "Department", "DateOfJoining",
    "AnnualLeaveQuota", "LeavesTaken", "LeavesRemaining", "LastLeaveDate",
]

# Active files (switched by use_rule_book / use_employee_db)
# Active files. None = not available yet (upload one from the app).
# The Streamlit app gives every browser visit its own copies and switches
# these with activate() before each run (see AGENT_LOCK).
RULE_BOOK_PATH = None
EMPLOYEE_DB_PATH = None
RULE_BOOK_IS_SAMPLE = False

# One agent run at a time, so two visitors never swap each other's files
AGENT_LOCK = threading.RLock()

# ============================================================
# 3. LOAD RULE BOOK
# ============================================================

def load_rule_book(path: str) -> str:
    """Read the complete rule book PDF."""

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


RULE_BOOK_TEXT = ""   # filled in below, once a rule book is found


# ============================================================
# 3b. SWITCH FILES (sample or uploaded)
# ============================================================

def check_employee_db(path: str):
    """Return an error message if the Excel file can't be used, else None."""
    try:
        df = pd.read_excel(path)
    except Exception as e:
        return f"Could not read the Excel file: {e}"
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return "Missing column(s): " + ", ".join(missing)
    if df.empty:
        return "The employee database has no rows."
    for col in ("AnnualLeaveQuota", "LeavesTaken", "LeavesRemaining"):
        if not pd.to_numeric(df[col], errors="coerce").notna().all():
            return f"Column {col} must contain only numbers."
    return None


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def is_sample_rule_book(path: str) -> bool:
    """True if this PDF is byte-for-byte the sample rule book."""
    return (os.path.exists(DEFAULT_RULE_BOOK_PATH)
            and _file_hash(path) == _file_hash(DEFAULT_RULE_BOOK_PATH))


def use_rule_book(path: str):
    """Make the agent follow this rule book PDF from now on."""
    text = load_rule_book(path)
    if not text.strip():
        raise ValueError("No text found in the PDF (is it a scanned image?).")
    activate(path, text, is_sample_rule_book(path), EMPLOYEE_DB_PATH)


def use_employee_db(path: str):
    """Make the agent read and update this Excel database from now on."""
    error = check_employee_db(path)
    if error:
        raise ValueError(error)
    activate(RULE_BOOK_PATH, RULE_BOOK_TEXT, RULE_BOOK_IS_SAMPLE, path)


def activate(rule_book_path, rule_book_text, rule_book_is_sample, employee_db_path):
    """Point the agent at a set of files (already checked) without re-reading them."""
    global RULE_BOOK_PATH, RULE_BOOK_TEXT, RULE_BOOK_IS_SAMPLE, EMPLOYEE_DB_PATH
    RULE_BOOK_PATH = rule_book_path
    RULE_BOOK_TEXT = rule_book_text or ""
    RULE_BOOK_IS_SAMPLE = bool(rule_book_is_sample)
    EMPLOYEE_DB_PATH = employee_db_path


def is_default_rule_book() -> bool:
    return RULE_BOOK_PATH is not None and RULE_BOOK_IS_SAMPLE


def files_ready() -> bool:
    """True when both a rule book and an employee database are loaded."""
    return RULE_BOOK_PATH is not None and EMPLOYEE_DB_PATH is not None


def load_startup_files():
    """Terminal version: use the sample files in the project folder, if present."""
    if os.path.exists(DEFAULT_RULE_BOOK_PATH):
        try:
            use_rule_book(DEFAULT_RULE_BOOK_PATH)
        except Exception as e:
            print(f"Skipping rule book: {e}")
    if os.path.exists(DEFAULT_EMPLOYEE_DB_PATH):
        try:
            use_employee_db(DEFAULT_EMPLOYEE_DB_PATH)
        except Exception as e:
            print(f"Skipping employee database: {e}")


load_startup_files()

#probabtion validation 

def validate_leave_request(employee, leave_date, number_of_days, is_medical_emergency=False):
    """
    Check whether an employee is eligible for the requested
    number of paid leave days.

    Returns an error message if the request fails.
    Returns None if these checks pass.
    """

    today = pd.Timestamp.today().normalize()

    # 0a. Basic sanity check on the number of days
    if number_of_days < 1:
        return "Leave rejected: number of days must be at least 1."

    # Checks 0b, 0c (same-day part) and 1-3 are the SAMPLE rule book's own
    # numbers. With an uploaded rule book they are skipped here and the
    # LLM enforces that rule book instead. Balance checks always run.
    sample_rules = is_default_rule_book()

    # 0b. Rule 4 - max 5 consecutive days through self-service
    if sample_rules and number_of_days > 5:
        return (
            "Leave rejected: more than 5 consecutive days needs direct "
            "manager approval outside this system."
        )

    # 0c. Rule 3 - advance notice (at least 1 day before, same day only
    #     for medical emergencies, never in the past)
    requested = pd.to_datetime(leave_date, format="%Y-%m-%d", errors="coerce")
    if pd.isna(requested):
        return "Leave rejected: leave_date must be in YYYY-MM-DD format."
    if requested < today:
        return "Leave rejected: leave date is in the past."
    if sample_rules and requested == today and not is_medical_emergency:
        return (
            "Leave rejected: leave must be applied at least 1 day in advance "
            "(same-day leave is only allowed for medical emergencies)."
        )

    # 1-3. Rule 2 - probation (sample rule book: 6 months)
    if sample_rules:
        joining_date = pd.to_datetime(
            employee["DateOfJoining"],
            errors="coerce"
        )

        # We cannot check probation without a valid joining date
        if pd.isna(joining_date):
            return "Cannot verify probation: joining date is missing or invalid."

        six_months_ago = today - pd.DateOffset(months=6)

        if joining_date > six_months_ago:
            return "Leave rejected: employee is still within the 6-month probation period."

    # 4. Get the employee's remaining leave balance
    remaining_days = int(employee["LeavesRemaining"])

    # 5. Reject if no paid leave is available
    if remaining_days <= 0:
        return "Leave rejected: no paid leave balance remains."

    # 6. Reject if the request exceeds the available balance
    if number_of_days > remaining_days:
        return (
            f"Leave rejected: requested {number_of_days} days, "
            f"but only {remaining_days} days remain."
        )

    # All checks in this function passed
    return None



# ============================================================
# 4. TOOL 1 — GET EMPLOYEE INFORMATION
# ============================================================

@tool
def get_employee_info(employee_id: str) -> str:
    """
    Look up an employee's information from the employee database.

    Use this tool whenever you need:
    - employee name
    - department
    - joining date
    - leave balance
    - leaves already taken
    """

    if EMPLOYEE_DB_PATH is None:
        return "No employee database is loaded yet. Please upload one first."
    df = pd.read_excel(EMPLOYEE_DB_PATH)

    # Make EmployeeID comparison case-insensitive
    mask = (
        df["EmployeeID"]
        .astype(str)
        .str.upper()
        == employee_id.upper()
    )

    if not mask.any():
        return f"No employee found with ID {employee_id}."

    row = df[mask].iloc[0]

    return (
        f"EmployeeID: {row['EmployeeID']}\n"
        f"Name: {row['Name']}\n"
        f"Department: {row['Department']}\n"
        f"DateOfJoining: {row['DateOfJoining']}\n"
        f"AnnualLeaveQuota: {row['AnnualLeaveQuota']}\n"
        f"LeavesTaken: {row['LeavesTaken']}\n"
        f"LeavesRemaining: {row['LeavesRemaining']}\n"
        f"LastLeaveDate: {row['LastLeaveDate']}"
    )



# ============================================================
# 5. TOOL 2 — APPLY LEAVE
# ============================================================

@tool
def apply_leave(
    employee_id: str,
    leave_date: str,
    number_of_days: int,
    is_medical_emergency: bool = False
) -> str:
    """
    Apply leave for an employee and update the Excel database.

    Only call this AFTER confirming the request against the rule book
    and the employee's balance. The tool re-checks every rule as a
    safety net before writing anything.
    leave_date must be in YYYY-MM-DD format.
    Set is_medical_emergency=True only if the employee says the leave
    is for a medical emergency (allows same-day leave).
    """

    # 1. Read the employee database
    if EMPLOYEE_DB_PATH is None:
        return "No employee database is loaded yet. Please upload one first."
    df = pd.read_excel(EMPLOYEE_DB_PATH)

    # 2. Find the employee using their ID
    mask = (
        df["EmployeeID"]
        .astype(str)
        .str.upper()
        == employee_id.upper()
    )

    # 3. Stop if employee does not exist
    if not mask.any():
        return f"No employee found with ID {employee_id}."

    # 4. Get the employee's row index
    idx = df[mask].index[0]

    # 5. Get the employee's complete record
    employee = df.loc[idx].copy()

    # 6. Validate the request against every rule in the rule book
    error = validate_leave_request(
        employee,
        leave_date,
        number_of_days,
        is_medical_emergency
    )

    # 7. Stop if any validation check fails
    if error:
        return error

    # 8. Get the current remaining balance
    remaining = int(employee["LeavesRemaining"])

    # 9. Update leave information
    df.loc[idx, "LeavesTaken"] = (
        int(employee["LeavesTaken"]) + number_of_days
    )

    df.loc[idx, "LeavesRemaining"] = (
        remaining - number_of_days
    )

    df.loc[idx, "LastLeaveDate"] = leave_date

    # 10. Save the updated database
    df.to_excel(
        EMPLOYEE_DB_PATH,
        index=False
    )

    # 11. Return confirmation
    new_balance = remaining - number_of_days

    return (
        f"Leave applied successfully for {employee_id} "
        f"on {leave_date} "
        f"for {number_of_days} day(s).\n"
        f"New remaining balance: {new_balance} day(s)."
    )


# ============================================================
# 6. REGISTER TOOLS
# ============================================================

tools = [
    get_employee_info,
    apply_leave
]


# ============================================================
# 7. CREATE GEMINI MODEL
# ============================================================

# Override with GEMINI_MODEL in .env if this model name is unavailable
# SPEED: leave checks are simple, so "low" thinking is enough and much faster.
# Set GEMINI_THINKING=default in .env to use the model's normal thinking.
THINKING = os.getenv("GEMINI_THINKING", "low")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    temperature=0,
    **({} if THINKING == "default" else {"thinking_level": THINKING}),
)


# ============================================================
# 8. GIVE TOOLS TO GEMINI
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 9. AGENT STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 10. SYSTEM PROMPT
# ============================================================

def build_system_prompt() -> str:
    """Built on every call so it always has today's date and the
    currently active (possibly uploaded) rule book."""
    return f"""
You are an HR Leave Assistant.

Today's date is {pd.Timestamp.today():%Y-%m-%d (%A)}. Use it to resolve
relative dates like "tomorrow" or "next Monday" and to check the
advance-notice rule. Always pass dates to tools as YYYY-MM-DD.

Your job is to help employees check and apply for leave.

You MUST follow the company rule book.

================ COMPANY RULE BOOK ================

{RULE_BOOK_TEXT}

============== END COMPANY RULE BOOK ==============


IMPORTANT RULES:

1. Always identify the employee before processing leave.

2. If the employee's record is given under "EMPLOYEE RECORDS" below,
   use it directly (it is live from the database). Otherwise call
   get_employee_info first.

3. Check the employee's current leave balance.

4. Check the employee's request against the company rule book.

5. Never approve leave that violates a company rule.

6. Only call apply_leave when you are sure that:
   - the employee exists
   - the employee has enough leave balance
   - the request follows the company rules

7. If the leave is not allowed:
   - DO NOT call apply_leave
   - explain clearly why the leave cannot be approved

8. If the leave is allowed:
   - call apply_leave
   - tell the employee that the leave was successfully applied

9. Never invent employee information.

10. Never invent company rules.

11. If important information is missing, ask the employee for it.
"""


# ============================================================
# 11. AGENT NODE
# ============================================================

# SPEED: only the last few messages are sent to the LLM
MAX_HISTORY = 12


def trim_history(messages: list) -> list:
    """Keep the last MAX_HISTORY messages, starting at a user message
    so a tool call is never separated from its tool result."""
    recent = messages[-MAX_HISTORY:]
    for i, m in enumerate(recent):
        if isinstance(m, HumanMessage):
            return recent[i:]
    return recent


def prefetch_employee_records(messages: list) -> str:
    """SPEED: if the user mentioned an employee ID, look the record up in
    code and hand it to the LLM, so it doesn't need a get_employee_info call.
    Read fresh on every step, so it is correct even after apply_leave."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            ids = faq.find_employee_ids(message_text(m), EMPLOYEE_DB_PATH)
            if ids and EMPLOYEE_DB_PATH:
                records = [get_employee_info.func(i) for i in ids[:3]]
                return "\n\n".join(records)
    return ""


def turn_used_tools(messages: list) -> bool:
    """Did the agent call any tool while answering the latest user message?"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return False
        if getattr(m, "tool_calls", None):
            return True
    return False


def message_text(message) -> str:
    """Gemini sometimes returns content as a list of parts - flatten it."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return content or ""


def agent_node(state: AgentState):

    messages = trim_history(state["messages"])

    system = build_system_prompt()
    records = prefetch_employee_records(messages)
    if records:
        system += (
            "\n\n================ EMPLOYEE RECORDS (live) ================\n"
            f"{records}\n"
        )

    full_messages = [
        {
            "role": "system",
            "content": system
        }
    ] + messages

    response = llm_with_tools.invoke(full_messages)

    return {
        "messages": [response]
    }


# ============================================================
# 12. DECIDE WHETHER TO USE A TOOL
# ============================================================

def should_continue(state: AgentState) -> str:

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# ============================================================
# 13. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# Starting point
graph.set_entry_point("agent")

# Agent → Tool or END
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Tool → Agent
graph.add_edge("tools", "agent")


# Compile graph
app = graph.compile()


# ============================================================
# 14. CHAT LOOP
# ============================================================

if __name__ == "__main__":

    print("\n====================================")
    print("     HR LEAVE ASSISTANT")
    print("====================================")
    print("Type 'exit' to quit.\n")

    if not files_ready():
        print("No rule book / employee database found.")
        print("Put the files next to agent.py, or upload them in the web app (streamlit run app.py).")
        raise SystemExit(1)

    chat_history = []

    while True:

        user_input = input("Employee: ")

        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break

        # FAST PATH: balance check / saved FAQ answer, no LLM call
        quick = faq.quick_answer(user_input, RULE_BOOK_PATH, EMPLOYEE_DB_PATH)
        if quick:
            answer, source = quick
            chat_history += [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ]
            print(f"\nAgent (instant, from {source}): {answer}\n")
            continue

        is_first_message = not chat_history
        chat_history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        try:

            result = app.invoke(
                {
                    "messages": chat_history
                }
            )

            chat_history = result["messages"]

            final_reply = chat_history[-1].content

            if isinstance(final_reply, list):
                final_reply = "".join(
                    item.get("text", "")
                    for item in final_reply
                    if isinstance(item, dict) and item.get("type") == "text"
                )

            print(f"\nAgent: {final_reply}\n")

            # Save general questions so next time they are instant
            faq.remember(user_input, final_reply, turn_used_tools(chat_history),
                         is_first_message, RULE_BOOK_PATH, EMPLOYEE_DB_PATH)

        except Exception as e:

            print(f"\nError: {e}\n")