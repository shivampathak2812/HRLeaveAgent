"""
Fast answers that skip the LLM completely.

1. Balance check  - "How many leaves do I have left? I'm E108"
   Answered straight from the Excel database (always live, never cached).

2. FAQ cache      - general policy questions like "What is the probation period?"
   Answers are saved in faq_cache.json. The same (or a very similar) question
   is answered from the file next time, with no LLM call.

Anything that applies leave, mentions a date, or depends on a specific
employee is NEVER cached - those always go to the agent.
"""

import hashlib
import json
import os
import re
from difflib import SequenceMatcher

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_CACHE_PATH = os.path.join(BASE_DIR, "faq_cache.json")

EMPLOYEE_ID_RE = re.compile(r"\b[eE]\s?-?\d{3,}\b")

# Words that mean the message is about a specific request or a date.
# Such messages must always go to the agent and are never cached.
TIME_SENSITIVE_WORDS = {
    "today", "tomorrow", "yesterday", "tonight", "next", "this", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "that", "why", "again",
}
DATE_RE = re.compile(r"\d{1,4}[-/.]\d{1,2}([-/.]\d{1,4})?|\b\d{1,2}(st|nd|rd|th)\b")

BALANCE_WORDS = {"balance", "left", "remaining", "remain", "how many", "available"}
REQUEST_WORDS = {"want", "apply", "take", "need", "book", "request", "off", "on"}

STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "i", "me", "my", "we", "you",
    "to", "of", "for", "in", "can", "what", "whats", "how", "please", "tell",
    "about", "there", "any", "be", "it", "and", "or", "per", "hi", "hello", "hey",
    "on", "at", "with", "if", "when", "will", "should", "much", "many", "while",
    "during", "get", "am", "allowed", "possible", "s",
}
# Words almost every leave question has - they don't tell questions apart
GENERIC_WORDS = {"leave", "day", "take", "apply", "employee", "company", "rule", "policy"}

MATCH_THRESHOLD = 0.82   # how similar a question must be to reuse a saved answer
MIN_WORDS_TO_CACHE = 3   # don't cache tiny follow-ups like "ok" or "and E102?"


# ------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------
def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(text.split())


def _keywords(text: str) -> set[str]:
    words = set(_normalize(text).split()) - STOPWORDS
    # crude singular form so "leaves" == "leave", "days" == "day"
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words}


def _similarity(a: str, b: str) -> float:
    ka, kb = _keywords(a), _keywords(b)
    if not ka or not kb:
        return 0.0
    jaccard = len(ka & kb) / len(ka | kb)
    seq = SequenceMatcher(None, " ".join(sorted(ka)), " ".join(sorted(kb))).ratio()
    # every meaningful word of the question appears in the saved question
    specific = ka - GENERIC_WORDS
    containment = len(specific & kb) / len(specific) if specific else 0.0
    return max(jaccard, seq, containment)


def _is_time_sensitive(text: str) -> bool:
    words = set(_normalize(text).split())
    return bool(words & TIME_SENSITIVE_WORDS) or bool(DATE_RE.search(text))


def _known_ids(employee_db_path: str) -> set[str]:
    try:
        df = pd.read_excel(employee_db_path, usecols=["EmployeeID"])
        return set(df["EmployeeID"].astype(str).str.strip().str.upper())
    except Exception:
        return set()


def find_employee_ids(text: str, employee_db_path: str | None = None) -> list[str]:
    """Employee IDs in a message: anything shaped like E101, plus any ID that
    exists in the database (so uploaded DBs with IDs like EMP001 or 1001 work)."""
    ids = [re.sub(r"[\s-]", "", m).upper() for m in EMPLOYEE_ID_RE.findall(text)]
    if employee_db_path:
        known = _known_ids(employee_db_path)
        tokens = re.findall(r"[A-Za-z0-9_-]+", text.upper())
        ids = [t for t in tokens if t in known] + ids
    return list(dict.fromkeys(ids))  # unique, keep order


# ------------------------------------------------------------
# FAQ cache file
# ------------------------------------------------------------
def _rule_book_hash(rule_book_path: str) -> str:
    """Fingerprint of the rule book PDF file itself (same on every computer)."""
    if not rule_book_path or not os.path.exists(rule_book_path):
        return "no-rule-book"
    with open(rule_book_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _read_file() -> dict:
    """faq_cache.json holds one list of answers per rule book:
    {"caches": {"<rule book hash>": {"entries": [...]}}}"""
    try:
        with open(FAQ_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"caches": {}}
    if "caches" not in data:  # old single-rule-book format
        data = {"caches": {data.get("rule_book_hash", ""): {"entries": data.get("entries", [])}}}
    return data


def _load_cache(rule_book_path: str) -> dict:
    """Saved answers for this rule book only."""
    current = _rule_book_hash(rule_book_path)
    entries = _read_file()["caches"].get(current, {}).get("entries", [])
    return {"rule_book_hash": current, "entries": entries}


def _save_cache(data: dict):
    """Save one rule book's answers, keeping the other rule books' answers."""
    full = _read_file()
    full["caches"][data["rule_book_hash"]] = {"entries": data["entries"]}
    with open(FAQ_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)


def _best_match(question: str, entries: list[dict]):
    best, best_score = None, 0.0
    for entry in entries:
        for q in entry["questions"]:
            score = _similarity(question, q)
            if score > best_score:
                best, best_score = entry, score
    return (best, best_score) if best_score >= MATCH_THRESHOLD else (None, best_score)


# ------------------------------------------------------------
# Balance check straight from the database
# ------------------------------------------------------------
def _balance_answer(message: str, employee_db_path: str | None, allowed_id: str | None = None):
    if not employee_db_path:
        return None
    ids = find_employee_ids(message, employee_db_path)
    text = _normalize(message)
    words = set(text.split())

    asks_balance = any(w in text for w in BALANCE_WORDS)
    is_request = bool(words & REQUEST_WORDS) or bool(DATE_RE.search(message))
    if allowed_id:
        # Logged-in employee: only ever their own balance. Another ID in the
        # message goes to the agent, which refuses it.
        if ids and ids != [allowed_id.upper()]:
            return None
        # without an ID, "how many leaves per year" is a policy question, so
        # only clear balance words count here
        if not ids and not words & {"left", "remaining", "remain", "balance", "available"}:
            return None
        ids = [allowed_id.upper()]
    if len(ids) != 1 or not asks_balance or is_request:
        return None

    df = pd.read_excel(employee_db_path)
    row = df[df["EmployeeID"].astype(str).str.upper() == ids[0]]
    if row.empty:
        return f"I couldn't find an employee with ID {ids[0]}. Please check the ID."
    r = row.iloc[0]
    last = r["LastLeaveDate"] if pd.notna(r["LastLeaveDate"]) else "none yet"
    return (
        f"Hi {r['Name']}! You have **{int(r['LeavesRemaining'])} leave day(s) remaining** "
        f"out of {int(r['AnnualLeaveQuota'])} this year "
        f"({int(r['LeavesTaken'])} taken so far, last leave: {last})."
    )


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def quick_answer(message: str, rule_book_path: str | None, employee_db_path: str | None,
                 allowed_id: str | None = None):
    """
    Try to answer without the LLM.
    Returns (answer, source) where source is "database" or "faq",
    or None if the message must go to the agent.
    """
    if not rule_book_path:   # no rule book yet - nothing to answer from
        return None
    balance = _balance_answer(message, employee_db_path, allowed_id)
    if balance:
        return balance, "database"

    if find_employee_ids(message, employee_db_path) or _is_time_sensitive(message):
        return None

    entry, _ = _best_match(message, _load_cache(rule_book_path)["entries"])
    if entry:
        return entry["answer"], "faq"
    return None


def remember(message: str, answer: str, used_tools: bool, is_first_message: bool,
             rule_book_path: str, employee_db_path: str | None = None):
    """
    Save an LLM answer to the FAQ cache, but only if it's a general question:
    first message of the chat (no context needed), no tools used, no employee ID,
    no dates, and long enough to be a real question.
    """
    if (
        used_tools
        or not is_first_message
        or not answer.strip()
        or find_employee_ids(message, employee_db_path)
        or _is_time_sensitive(message)
        or len(_normalize(message).split()) < MIN_WORDS_TO_CACHE
    ):
        return False

    data = _load_cache(rule_book_path)
    entry, _ = _best_match(message, data["entries"])
    if entry:  # similar question already saved - just add this wording
        if message not in entry["questions"]:
            entry["questions"].append(message)
    else:
        data["entries"].append({"questions": [message], "answer": answer})
    _save_cache(data)
    return True


def clear_cache(rule_book_path: str):
    """Forget the saved answers for this rule book."""
    _save_cache({"rule_book_hash": _rule_book_hash(rule_book_path), "entries": []})