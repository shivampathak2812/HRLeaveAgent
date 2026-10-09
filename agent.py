import os
import pandas as pd

from dotenv import load_dotenv
from pypdf import PdfReader
from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file")


# ============================================================
# 2. FILE PATHS
# ============================================================

RULE_BOOK_PATH = "DIY - HR - Rule Book.pdf"
EMPLOYEE_DB_PATH = "DIY - HR - Employee DB.xlsx"

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


RULE_BOOK_TEXT = load_rule_book(RULE_BOOK_PATH)

#probabtion validation 

def validate_probation_and_balance(employee, number_of_days):
    """
    Check whether an employee is eligible for the requested
    number of paid leave days.

    Returns an error message if the request fails.
    Returns None if these checks pass.
    """

    # 1. Get the employee's joining date
    joining_date = pd.to_datetime(
        employee["DateOfJoining"],
        errors="coerce"
    )

    # We cannot check probation without a valid joining date
    if pd.isna(joining_date):
        return "Cannot verify probation: joining date is missing or invalid."

    # 2. Calculate the date six months before today
    today = pd.Timestamp.today().normalize()
    six_months_ago = today - pd.DateOffset(months=6)

    # 3. Reject employees who joined less than six months ago
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
    number_of_days: int
) -> str:
    """
    Apply leave for an employee and update the Excel database.

    Validate probation and leave balance before updating Excel.
    leave_date must be in YYYY-MM-DD format.
    """

    # 1. Read the employee database
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

    # 6. Validate probation and leave balance
    error = validate_probation_and_balance(
        employee,
        number_of_days
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

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
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

SYSTEM_PROMPT = f"""
You are an HR Leave Assistant.

Your job is to help employees check and apply for leave.

You MUST follow the company rule book.

================ COMPANY RULE BOOK ================

{RULE_BOOK_TEXT}

============== END COMPANY RULE BOOK ==============


IMPORTANT RULES:

1. Always identify the employee before processing leave.

2. ALWAYS call get_employee_info first.

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

def agent_node(state: AgentState):

    messages = state["messages"]

    full_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
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

    chat_history = []

    while True:

        user_input = input("Employee: ")

        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break

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

        except Exception as e:

            print(f"\nError: {e}\n")