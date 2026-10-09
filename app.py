"""
Streamlit UI for the HR Leave Agent.

Run:  streamlit run app.py
"""

import io
import os
import shutil

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

st.set_page_config(
    page_title="HR Leave Assistant",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Load the agent (shows a friendly error instead of a stack trace)
# ------------------------------------------------------------
try:
    import agent
    import faq
except Exception as e:  # e.g. missing GOOGLE_API_KEY or missing files
    st.error(f"Could not start the agent: {e}")
    st.info("Add `GOOGLE_API_KEY=your-key` to a `.env` file next to app.py and restart.")
    st.stop()


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a8f 100%);
        border-radius: 16px; padding: 1.4rem 1.6rem; color: #fff; margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 1.7rem; color: #fff; }
    .hero p  { margin: .35rem 0 0; opacity: .85; font-size: .95rem; }
    .chip {
        display: inline-block; padding: .15rem .6rem; border-radius: 999px;
        font-size: .75rem; font-weight: 600; margin-top: .3rem;
    }
    .chip-ok    { background: #d1fae5; color: #065f46; }
    .chip-warn  { background: #fef3c7; color: #92400e; }
    .chip-bad   { background: #fee2e2; color: #991b1b; }
    .chip-fast  { background: #e0f2fe; color: #075985; margin: 0 0 .4rem; }
    .emp-name { font-size: 1.05rem; font-weight: 700; margin-bottom: 0; }
    .emp-meta { font-size: .8rem; opacity: .7; margin-top: 0; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_db() -> pd.DataFrame:
    """Read this visit's employee database fresh every time (it changes after approvals)."""
    df = pd.read_excel(ws["db_path"])
    df["DateOfJoining"] = pd.to_datetime(df["DateOfJoining"], errors="coerce")
    return df


SOURCE_BADGES = {
    "database": "<span class='chip chip-fast'>⚡ Instant · from database</span>",
    "faq": "<span class='chip chip-fast'>⚡ Instant · saved FAQ answer</span>",
}


def employee_status(row) -> tuple[str, str]:
    """Return (label, css class) for the status chip."""
    six_months_ago = pd.Timestamp.today().normalize() - pd.DateOffset(months=6)
    if pd.notna(row["DateOfJoining"]) and row["DateOfJoining"] > six_months_ago:
        return "On probation", "chip-warn"
    if int(row["LeavesRemaining"]) <= 0:
        return "No balance left", "chip-bad"
    return "Eligible", "chip-ok"


def describe_tool_call(call: dict) -> str:
    args = call.get("args", {})
    if call["name"] == "get_employee_info":
        return f"🔍 Looking up employee **{args.get('employee_id', '?')}**"
    if call["name"] == "apply_leave":
        extra = " (medical emergency)" if args.get("is_medical_emergency") else ""
        return (
            f"📝 Applying **{args.get('number_of_days', '?')} day(s)** for "
            f"**{args.get('employee_id', '?')}** from **{args.get('leave_date', '?')}**{extra}"
        )
    return f"🛠️ Calling `{call['name']}`"


def render_steps(steps: list[dict]):
    """Show the tool calls the agent made, inside a collapsible box."""
    if not steps:
        return
    with st.expander(f"Agent steps ({len(steps)})", expanded=False):
        for step in steps:
            st.markdown(step["label"])
            st.code(step["result"], language=None)


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []      # messages sent to LangGraph
if "display" not in st.session_state:
    st.session_state.display = []      # what we render: {role, content, steps}
if "pending" not in st.session_state:
    st.session_state.pending = None    # prompt queued by a button click
if "upload_round" not in st.session_state:
    st.session_state.upload_round = 0  # bump to clear the file uploaders


# ------------------------------------------------------------
# Active files (shared by everyone, saved on disk)
# - Rule book / database: an uploaded file if there is one, else the sample file.
# - Approved leaves are written to the active database permanently (rule 7).
# - A copy of each database as it was first loaded is kept in .originals/,
#   so "Restore original data" can undo test leaves.
# - Refreshing the page only starts a new chat; the data stays.
# ------------------------------------------------------------
UPLOAD_DIR = os.path.join(agent.BASE_DIR, "uploads")
UPLOADED_RULE_BOOK = os.path.join(UPLOAD_DIR, "rule_book.pdf")
UPLOADED_DB = os.path.join(UPLOAD_DIR, "employee_db.xlsx")
ORIGINALS_DIR = os.path.join(agent.BASE_DIR, ".originals")


def original_copy_path(db_path: str) -> str:
    name = "uploaded_employee_db.xlsx" if db_path == UPLOADED_DB else "sample_employee_db.xlsx"
    return os.path.join(ORIGINALS_DIR, name)


def keep_original(db_path: str, overwrite: bool = False):
    """Remember what the database looked like before any leave was applied."""
    os.makedirs(ORIGINALS_DIR, exist_ok=True)
    backup = original_copy_path(db_path)
    if overwrite or not os.path.exists(backup):
        shutil.copy(db_path, backup)


@st.cache_data(show_spinner=False)
def read_rule_book(path: str, modified: float) -> tuple[str, bool]:
    """Rule book text + whether it is the sample one (cached until the file changes)."""
    return agent.load_rule_book(path), agent.is_sample_rule_book(path)


def current_files() -> dict:
    rb = UPLOADED_RULE_BOOK if os.path.exists(UPLOADED_RULE_BOOK) else (
        agent.DEFAULT_RULE_BOOK_PATH if os.path.exists(agent.DEFAULT_RULE_BOOK_PATH) else None)
    db = UPLOADED_DB if os.path.exists(UPLOADED_DB) else (
        agent.DEFAULT_EMPLOYEE_DB_PATH if os.path.exists(agent.DEFAULT_EMPLOYEE_DB_PATH) else None)
    files = {"rb_path": rb, "rb_text": "", "rb_sample": False,
             "rb_source": None if rb is None else ("uploaded" if rb == UPLOADED_RULE_BOOK else "sample"),
             "db_path": db,
             "db_source": None if db is None else ("uploaded" if db == UPLOADED_DB else "sample")}
    if rb:
        files["rb_text"], files["rb_sample"] = read_rule_book(rb, os.path.getmtime(rb))
    if db:
        keep_original(db)
    return files


ws = current_files()


def activate_workspace():
    """Point the agent at the active files. Call only while holding agent.AGENT_LOCK."""
    agent.activate(ws["rb_path"], ws["rb_text"], ws["rb_sample"], ws["db_path"])


def db_download_button(key: str, label: str = "⬇️ Download updated database"):
    """Download the active Excel database exactly as it is saved right now."""
    with open(ws["db_path"], "rb") as f:
        data = f.read()
    name = "employee_database_updated.xlsx"
    st.download_button(
        label, data=data, file_name=name, key=key, width="stretch",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def remove_uploads():
    """Go back to the sample files."""
    with agent.AGENT_LOCK:
        for path in (UPLOADED_RULE_BOOK, UPLOADED_DB, original_copy_path(UPLOADED_DB)):
            if os.path.exists(path):
                os.remove(path)
    reset_chat()


def restore_original_data():
    """Undo every leave applied since the database was first loaded/uploaded."""
    with agent.AGENT_LOCK:
        shutil.copy(original_copy_path(ws["db_path"]), ws["db_path"])
    reset_chat()


def queue_prompt(text: str):
    st.session_state.pending = text


def reset_chat():
    st.session_state.history = []
    st.session_state.display = []
    st.session_state.pending = None


def template_excel() -> bytes:
    """Empty employee database with the right columns, for download."""
    sample = pd.DataFrame([{
        "EmployeeID": "E101", "Name": "Asha Rao", "Department": "Engineering",
        "Email": "asha.rao@company.com", "DateOfJoining": "2022-03-14",
        "AnnualLeaveQuota": 18, "LeavesTaken": 6, "LeavesRemaining": 12,
        "LastLeaveDate": "2026-08-20",
    }])
    buf = io.BytesIO()
    sample.to_excel(buf, index=False, sheet_name="Employees")
    return buf.getvalue()


def save_uploads(pdf_file, xlsx_file) -> list[str]:
    """Check uploaded files and save them as the active rule book / database.
    Returns a list of error messages (empty = success)."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    errors = []

    with agent.AGENT_LOCK:
        if xlsx_file is not None:
            tmp = UPLOADED_DB + ".tmp.xlsx"
            with open(tmp, "wb") as f:
                f.write(xlsx_file.getvalue())
            error = agent.check_employee_db(tmp)
            if error:
                os.remove(tmp)
                errors.append(f"Employee database: {error}")
            else:
                os.replace(tmp, UPLOADED_DB)
                keep_original(UPLOADED_DB, overwrite=True)  # new upload = new original

        if pdf_file is not None:
            tmp = UPLOADED_RULE_BOOK + ".tmp.pdf"
            with open(tmp, "wb") as f:
                f.write(pdf_file.getvalue())
            try:
                if not agent.load_rule_book(tmp).strip():
                    raise ValueError("no text found in the PDF (is it a scanned image?)")
                os.replace(tmp, UPLOADED_RULE_BOOK)
            except Exception as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                errors.append(f"Rule book: {e}")

    return errors


# ------------------------------------------------------------
# Sidebar - employee lookup, database, rule book
# ------------------------------------------------------------
ready = bool(ws["rb_path"] and ws["db_path"])   # both files available?
db = load_db() if ws["db_path"] else None
tomorrow = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")




with st.sidebar:
    # ---------------- Upload your own files ----------------
    has_uploads = "uploaded" in (ws["rb_source"], ws["db_source"])
    # open automatically when something is missing
    with st.expander("📤 Upload your files", expanded=not ready):
        r = st.session_state.upload_round
        xlsx = st.file_uploader("Employee database (.xlsx)", type=["xlsx"], key=f"xlsx{r}")
        pdf = st.file_uploader("Rule book (.pdf)", type=["pdf"], key=f"pdf{r}")
        st.download_button(
            "⬇️ Excel template (required columns)",
            data=template_excel(),
            file_name="employee_database_template.xlsx",
            width="stretch",
        )
        if st.button("✅ Use these files", disabled=not (xlsx or pdf),
                     type="primary", width="stretch"):
            errors = save_uploads(pdf, xlsx)
            if errors:
                for err in errors:
                    st.error(err)
            else:
                reset_chat()
                st.session_state.upload_round += 1
                st.toast("Files uploaded. The assistant now uses them.", icon="📤")
                st.rerun()

        sample_exists = os.path.exists(agent.DEFAULT_RULE_BOOK_PATH) or os.path.exists(
            agent.DEFAULT_EMPLOYEE_DB_PATH)
        remove_label = "↩️ Back to sample files" if sample_exists else "🗑️ Remove uploaded files"
        if has_uploads and st.button(remove_label, width="stretch"):
            remove_uploads()
            st.rerun()

        if ws["db_path"] and st.button("♻️ Restore original data", width="stretch",
                                       help="Undo all leaves applied since this database was loaded"):
            restore_original_data()
            st.toast("Database restored to its original values.", icon="♻️")
            st.rerun()

        rb = ws["rb_source"] or "missing"
        dbs = ws["db_source"] or "missing"
        st.caption(f"In use: rule book = **{rb}**, database = **{dbs}**")
        if rb == "uploaded":
            st.caption(
                "With your own rule book, the assistant follows its rules. Balance "
                "checks are always enforced in code."
            )

    if db is not None:
        st.markdown("### 👤 Employee")
        labels = {f"{r.EmployeeID} · {r.Name}": r.EmployeeID for r in db.itertuples()}
        choice = st.selectbox("Select employee", list(labels.keys()), label_visibility="collapsed")
        emp = db[db["EmployeeID"] == labels[choice]].iloc[0]

        status, css = employee_status(emp)
        joined = emp["DateOfJoining"].strftime("%d %b %Y") if pd.notna(emp["DateOfJoining"]) else "—"
        st.markdown(
            f"<p class='emp-name'>{emp['Name']}</p>"
            f"<p class='emp-meta'>{emp['Department']} · joined {joined}</p>"
            f"<span class='chip {css}'>{status}</span>",
            unsafe_allow_html=True,
        )

        quota = int(emp["AnnualLeaveQuota"]) or 1
        remaining = int(emp["LeavesRemaining"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Left", remaining)
        c2.metric("Taken", int(emp["LeavesTaken"]))
        c3.metric("Quota", quota)
        st.progress(min(max(remaining / quota, 0.0), 1.0), text=f"{remaining} of {quota} days left")

        last = emp["LastLeaveDate"]
        st.caption(f"Last leave: {last if pd.notna(last) else 'none yet'}")

        st.button(
            f"Ask for 1 day leave for {emp['EmployeeID']}",
            on_click=queue_prompt,
            args=(f"I'm employee {emp['EmployeeID']}, I want 1 day leave on {tomorrow}.",),
            width="stretch",
        )

        st.divider()

        with st.expander("📊 Employee database", expanded=False):
            view = db.copy()
            view["DateOfJoining"] = view["DateOfJoining"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                view[["EmployeeID", "Name", "Department", "LeavesTaken", "LeavesRemaining", "LastLeaveDate"]],
                hide_index=True,
                width="stretch",
            )

        db_download_button("dl_sidebar")
        saved_at = pd.Timestamp.fromtimestamp(os.path.getmtime(ws["db_path"]))
        st.caption(f"Database last saved: {saved_at:%d %b %Y, %I:%M:%S %p}")

    with st.expander("📘 Rule book", expanded=False):
        if ws["rb_text"].strip():
            st.text(ws["rb_text"].strip())
        else:
            st.caption("No rule book uploaded yet.")

    with st.expander("⚡ FAQ: click for an instant answer", expanded=ready):
        cache = faq._load_cache(ws["rb_path"])["entries"] if ws["rb_path"] else []
        if not cache:
            st.caption("Nothing saved yet. General questions get saved after the first answer.")
        for i, e in enumerate(cache):
            # clicking sends the question to the chat; it is answered from the
            # saved FAQ, so there is no LLM call
            st.button(
                e["questions"][0],
                key=f"faq{i}",
                on_click=queue_prompt,
                args=(e["questions"][0],),
                width="stretch",
            )
        if cache and st.button("🗑️ Clear saved answers", width="stretch"):
            faq.clear_cache(ws["rb_path"])
            st.rerun()

    st.divider()
    st.button("🔄 New conversation", on_click=reset_chat, width="stretch")
    st.caption("Approved leaves are saved to the database. Refresh starts a new chat.")
    st.caption(f"Today: {pd.Timestamp.today():%A, %d %b %Y}")


# ------------------------------------------------------------
# Main area
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🗓️ HR Leave Assistant</h1>
      <p>Ask for leave in plain English. The assistant checks your balance and the company
      rule book, then approves and updates the records, or tells you why it can't.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Setup needed - say exactly what is missing
if not ready:
    missing = []
    if not ws["db_path"]:
        missing.append("the **employee database** (.xlsx)")
    if not ws["rb_path"]:
        missing.append("the **rule book** (.pdf)")
    st.info(
        "👋 To get started, upload " + " and ".join(missing)
        + " from **📤 Upload your files** in the sidebar."
    )

# Empty state - example prompts
elif not st.session_state.display:
    st.markdown("**Try one of these:**")
    ids = db["EmployeeID"].astype(str).tolist()
    id1, id2 = ids[0], ids[min(1, len(ids) - 1)]
    examples = [
        f"I'm employee {id1}, I want 2 days leave on {tomorrow}",
        f"I'm {id2}, can I take leave tomorrow?",
        f"How many leaves do I have left? I'm {id1}",
        "What is the probation period rule?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        cols[i % 2].button(ex, key=f"ex{i}", on_click=queue_prompt, args=(ex,), width="stretch")

# Chat history
for i, msg in enumerate(st.session_state.display):
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
        if msg["role"] == "assistant":
            if msg.get("source") in SOURCE_BADGES:
                st.markdown(SOURCE_BADGES[msg["source"]], unsafe_allow_html=True)
            render_steps(msg.get("steps", []))
        st.markdown(msg["content"])
        if msg.get("leave_applied") and ws["db_path"]:
            db_download_button(f"dl_msg_{i}", "⬇️ Download the updated database")

# New input (typed or from a button)
typed = st.chat_input(
    "e.g. I'm employee E101, I want 2 days leave on 2026-10-20" if ready
    else "Upload the rule book and employee database first",
    disabled=not ready,
)
prompt = (typed or st.session_state.pending) if ready else None
st.session_state.pending = None

if prompt:
    st.session_state.display.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    steps: list[dict] = []
    leave_applied = False
    source = "agent"

    with st.chat_message("assistant", avatar="🤖"):

        # ---- FAST PATH: balance from DB or saved FAQ answer (no LLM call) ----
        quick = faq.quick_answer(prompt, ws["rb_path"], ws["db_path"])

        if quick:
            reply, source = quick
            st.markdown(SOURCE_BADGES[source], unsafe_allow_html=True)
            st.markdown(reply)
            st.session_state.history += [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply},
            ]

        # ---- AGENT PATH: stream tool steps and the reply token by token ----
        else:
            is_first_message = not st.session_state.history
            history = st.session_state.history + [{"role": "user", "content": prompt}]
            pending_labels: dict[str, str] = {}
            status_box = st.status("Thinking…", expanded=True)
            reply_box = st.empty()
            streamed = ""
            seen = len(history)
            final_state = None
            reply = ""

            # Lock so another visitor's run can't switch the files mid-way
            agent.AGENT_LOCK.acquire()
            try:
                activate_workspace()
                for mode, chunk in agent.app.stream(
                    {"messages": history}, stream_mode=["messages", "values"]
                ):
                    if mode == "messages":
                        # token-by-token text from the LLM
                        token, meta = chunk
                        if meta.get("langgraph_node") == "agent" and not getattr(
                            token, "tool_call_chunks", None
                        ):
                            streamed += agent.message_text(token)
                            if streamed:
                                reply_box.markdown(streamed + " ▌")
                        continue

                    # mode == "values": full state after each node
                    final_state = chunk
                    new_messages = chunk["messages"][seen:]
                    seen = len(chunk["messages"])
                    for m in new_messages:
                        if isinstance(m, AIMessage) and m.tool_calls:
                            streamed = ""          # text before a tool call isn't the answer
                            reply_box.empty()
                            for call in m.tool_calls:
                                label = describe_tool_call(call)
                                pending_labels[call["id"]] = label
                                status_box.markdown(label)
                        elif isinstance(m, ToolMessage):
                            result = agent.message_text(m)
                            steps.append({
                                "label": pending_labels.get(m.tool_call_id, f"🛠️ {m.name}"),
                                "result": result,
                            })
                            if m.name == "apply_leave" and result.startswith("Leave applied"):
                                leave_applied = True

                status_box.update(
                    label=f"Done · {len(steps)} step(s)" if steps else "Done",
                    state="complete",
                    expanded=False,
                )
                st.session_state.history = final_state["messages"]
                reply = agent.message_text(final_state["messages"][-1]) or "_(no reply)_"
                reply_box.markdown(reply)

                # Save general questions so next time they are instant
                faq.remember(prompt, reply, agent.turn_used_tools(final_state["messages"]),
                             is_first_message, ws["rb_path"], ws["db_path"])

            except Exception as e:
                status_box.update(label="Failed", state="error", expanded=False)
                reply = f"⚠️ Something went wrong: `{e}`"
                reply_box.error(reply)
            finally:
                agent.AGENT_LOCK.release()

    st.session_state.display.append(
        {"role": "assistant", "content": reply, "steps": steps, "source": source,
         "leave_applied": leave_applied}
    )

    if leave_applied:
        st.toast("Leave approved and database updated", icon="✅")
        st.rerun()  # refresh the sidebar with the new balance