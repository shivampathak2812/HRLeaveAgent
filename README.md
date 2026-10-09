# HR Leave Agent

A chat assistant that lets an employee apply for leave in plain English. It looks up the
employee in the database, checks the request against the company rule book, and either
approves it (and updates the database) or explains why not.

Built with Python, LangGraph and Gemini (`agent.py`), with a Streamlit web UI (`app.py`).

Example message: `I'm employee E101, I want 2 days leave on 2026-10-20`

---

## Setup and run

```bash
uv sync                     # or: pip install -e .
cp .env.example .env        # then put your Gemini key in .env
uv run agent.py             # terminal chat   (or: python agent.py)
uv run streamlit run app.py # web UI          (or: streamlit run app.py)
```

### Login and privacy
- **Employee login:** Employee ID + the email in the database. An employee sees only their own balance, the rule book and the FAQ, never the employee list, uploads or downloads.
- **The agent is locked to the logged-in employee:** both tools refuse any other ID (`access_denied()` in `agent.py`), the prompt says who is logged in, and the balance shortcut only answers for that employee. Typing "I'm E108" does not work.
- **HR login:** password from `HR_ADMIN_PASSWORD` (`.env` locally, Secrets on Streamlit Cloud). HR gets the full view: every employee, database table, upload, restore and download, and can apply leave for anyone.
- Without `HR_ADMIN_PASSWORD` the HR login is off. Without an `Email` column employees can't log in.

### Streamlit UI (`app.py`)
- Chat with the agent; each tool call (employee lookup, leave apply) shows live, and you can expand it to see the result.
- Employee sidebar: own balance, status (eligible / probation / no balance), a one-click leave request, rule book, FAQ.
- HR sidebar: pick any employee, full database table, download, rule book, FAQ; the balance refreshes after every approval.
- Approved leaves are saved to the Excel file; refresh starts a new chat; **♻️ Restore original data** undoes test leaves.
- "New conversation" clears the chat memory.

### Use your own rule book and database
HR login → sidebar → **📤 Upload your files**:
- Upload an Excel database and/or a rule book PDF, then click **Use these files**.
- The Excel file must have these columns: `EmployeeID, Name, Department, DateOfJoining, AnnualLeaveQuota, LeavesTaken, LeavesRemaining, LastLeaveDate`. Download the template from the same panel.
- Files are checked first (missing columns, non-numeric balances, PDF with no text) and rejected with a clear message.
- Uploaded files are saved in `uploads/` (git-ignored) and stay active until **Back to sample files**.
- Approved leaves are **saved permanently** to the active database (rule 7). Refreshing the page only starts a new chat.
- **♻️ Restore original data** undoes all test leaves: a copy of each database as it was first loaded/uploaded is kept in `.originals/` (git-ignored).
- Employee IDs can be any format (`E101`, `EMP001`, `1001`); they're matched against the database.
- With an uploaded rule book, the assistant follows its rules. The code still always enforces: employee exists, valid future date, enough balance. The sample-only checks (6-month probation, max 5 days, 1-day notice) are switched off because your rule book may have different numbers.
- Saved FAQ answers are stored per rule book (by file fingerprint), so an uploaded rule book gets its own FAQ list and the sample FAQ stays intact.
- The sample files are optional. If you delete them, the app opens with the upload panel and the chat stays disabled until both a rule book and a database are uploaded. (The terminal version `agent.py` needs the files in the folder.)

Graph: `agent` node (Gemini decides) ⇄ `tools` node (`get_employee_info`, `apply_leave`) until
the agent replies with no tool call.

`apply_leave` re-checks every rule before writing to the Excel file (safety net in case the LLM
reasons wrongly):

| Rule book rule | Enforced by |
|---|---|
| 1. Annual quota 18 days | DB `AnnualLeaveQuota` / `LeavesRemaining` |
| 2. Probation (< 6 months) | `validate_leave_request` |
| 3. Advance notice ≥ 1 day (same day only for medical emergency, no past dates) | `validate_leave_request` (`is_medical_emergency` flag) |
| 4. Max 5 consecutive days | `validate_leave_request` |
| 5. Insufficient balance → reject | `validate_leave_request` |
| 6. Zero balance → reject | `validate_leave_request` |
| 7. Deduct balance + update last leave date | `apply_leave` writes Excel |

Today's date is injected into the system prompt so "tomorrow" / "next Monday" resolve correctly.

---

## Speed

| What | How | Effect |
|---|---|---|
| Low thinking | `thinking_level="low"` on Gemini (`GEMINI_THINKING` in `.env`) | Each LLM call is faster |
| Pre-fetched employee record | If the message has an ID like `E101`, the record is read in code and put in the prompt | 2 LLM calls per leave request instead of 3 |
| Short history | Only the last 12 messages go to the LLM | Long chats stay fast |
| Streaming | Streamlit shows the reply word by word | Feels instant |
| Balance fast path | "How many leaves left? I'm E108" is answered from the Excel file | 0 LLM calls |
| FAQ cache (`faq.py`, `faq_cache.json`) | General policy questions are answered from saved answers | 0 LLM calls |

### FAQ cache rules
- `faq_cache.json` ships with answers for the common rule-book questions (quota, probation, notice, max days, balance, how to apply).
- New general questions answered by the LLM are saved automatically, but only if: it's the first message of the chat, no tool was used, there is no employee ID, and there is no date or day word ("tomorrow", "Monday", "2026-10-20").
- Similar wording matches too ("can I take leave while on probation" → the probation answer).
- Leave requests are **never** cached; they always go to the agent.
- Answers are stored per rule book, so a different rule book never gets the wrong answers.
- Clear them from the Streamlit sidebar ("Saved FAQ answers") or by deleting `faq_cache.json`.

---

## Test cases (against the sample DB)

| Message | Expected |
|---|---|
| E101, 2 days on a future date | Approved, 12 → 10 |
| E104, 1 day | Rejected — zero balance |
| How many leaves left? I'm E108 | Instant: 7 days (from database) |
| What is the probation period? | Instant (saved FAQ) |
| E107, 1 day | Rejected — probation (joined 2026-06-01) |
| E108, 8 days | Rejected — over 5 consecutive days |
| E102, 2 days today | Rejected — needs 1 day notice (unless medical emergency) |
| E999 | Employee not found |