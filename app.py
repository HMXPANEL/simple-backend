import os
import json
import re
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import uvicorn

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Android AI Agent Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODEL    = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions",
)
MAX_RETRIES     = int(os.environ.get("MAX_RETRIES", 2))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 30))

# ── Action Whitelist ──────────────────────────────────────────────────────────
ALLOWED_ACTIONS = {"open_app", "send_message", "click", "type", "wait"}

# Required fields per action (beyond "action" itself)
ACTION_REQUIRED_FIELDS: dict[str, list[str]] = {
    "open_app":      ["app"],
    "send_message":  ["app", "contact", "message"],
    "click":         [],          # text OR coordinates — checked separately
    "type":          ["text"],
    "wait":          ["duration"],
}

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Android AI agent.
Convert every user request into structured JSON.

Rules:
- ONLY return valid JSON
- No explanations, no extra text
- Always return a 'tasks' array
- Each task must be executable on Android

Allowed actions:
- open_app      → requires: app
- send_message  → requires: app, contact, message
- click         → requires: text OR coordinates
- type          → requires: text
- wait          → requires: duration (milliseconds, integer)

If multiple steps are needed, return multiple tasks in order."""

# ── In-Memory Conversation Store ──────────────────────────────────────────────
# Keyed by session_id (optional header). Falls back to a single shared slot.
_conversation_store: dict[str, list[dict]] = {}

MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", 6))  # user+assistant pairs


def get_history(session_id: str) -> list[dict]:
    return _conversation_store.get(session_id, [])


def push_history(session_id: str, role: str, content: str) -> None:
    history = _conversation_store.setdefault(session_id, [])
    history.append({"role": role, "content": content})
    # Keep only last N pairs (2 messages per turn)
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(history) > max_msgs:
        _conversation_store[session_id] = history[-max_msgs:]


def clear_history(session_id: str) -> None:
    _conversation_store.pop(session_id, None)


# ── Core Functions ────────────────────────────────────────────────────────────

def build_messages(session_id: str, user_message: str) -> list[dict]:
    """Assemble full message list: system prompt + history + new user turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_history(session_id))
    messages.append({"role": "user", "content": user_message})
    return messages


def call_nvidia_api(session_id: str, user_message: str) -> str | None:
    """
    Call NVIDIA NIM with retry logic.
    Retries up to MAX_RETRIES times on transient failures.
    Returns raw text content or None on failure.
    """
    if not NVIDIA_API_KEY:
        log.error("NVIDIA_API_KEY is not set")
        return None

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":       NVIDIA_MODEL,
        "messages":    build_messages(session_id, user_message),
        "temperature": 0.2,
        "max_tokens":  1024,
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("NVIDIA API call — attempt %d/%d", attempt, MAX_RETRIES)
            response = requests.post(
                NVIDIA_BASE_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            raw  = data["choices"][0]["message"]["content"].strip()
            log.info("NVIDIA raw response (attempt %d): %s", attempt, raw)
            return raw

        except requests.exceptions.Timeout as e:
            last_error = e
            log.warning("Attempt %d timed out", attempt)
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else "?"
            log.warning("Attempt %d HTTP error %s: %s", attempt, status, e)
            # Do not retry on client-side errors
            if e.response is not None and e.response.status_code < 500:
                break
        except (KeyError, IndexError, ValueError) as e:
            last_error = e
            log.warning("Attempt %d unexpected response structure: %s", attempt, e)
        except Exception as e:
            last_error = e
            log.warning("Attempt %d unexpected error: %s", attempt, e)

        if attempt < MAX_RETRIES:
            wait = attempt * 1.5
            log.info("Retrying in %.1fs…", wait)
            time.sleep(wait)

    log.error("All %d attempts failed. Last error: %s", MAX_RETRIES, last_error)
    return None


def extract_json(raw: str) -> dict | None:
    """
    Extract the first valid JSON object from a raw string.
    Handles markdown fences, leading/trailing garbage text.
    """
    if not raw:
        return None

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract first {...} block via regex
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    log.warning("Could not extract valid JSON from response")
    return None


def validate_tasks(data: dict) -> tuple[bool, str]:
    """
    Full validation of the tasks payload:
    - tasks key exists and is a non-empty list
    - every task has a whitelisted action
    - every task has its required fields
    - wait tasks have an integer duration
    - click tasks have text or coordinates
    """
    if not isinstance(data, dict):
        return False, "Response is not a JSON object"

    if "tasks" not in data:
        return False, "Missing 'tasks' key in response"

    if not isinstance(data["tasks"], list):
        return False, "'tasks' must be an array"

    if len(data["tasks"]) == 0:
        return False, "'tasks' array is empty"

    for i, task in enumerate(data["tasks"]):
        if not isinstance(task, dict):
            return False, f"Task[{i}] is not an object"

        # ── Action key present ────────────────────────────────────────────────
        if "action" not in task:
            return False, f"Task[{i}] missing 'action'"

        action = task["action"]

        # ── Action whitelist ──────────────────────────────────────────────────
        if action not in ALLOWED_ACTIONS:
            return False, f"Task[{i}] has invalid action '{action}'. Allowed: {sorted(ALLOWED_ACTIONS)}"

        # ── Required fields per action ────────────────────────────────────────
        required = ACTION_REQUIRED_FIELDS.get(action, [])
        for field in required:
            if field not in task:
                return False, f"Task[{i}] action '{action}' missing required field '{field}'"

        # ── wait: duration must be a positive integer ─────────────────────────
        if action == "wait":
            duration = task.get("duration")
            if not isinstance(duration, int) or duration <= 0:
                return False, f"Task[{i}] 'wait' requires 'duration' to be a positive integer (ms)"

        # ── click: must have text or coordinates ──────────────────────────────
        if action == "click":
            has_text   = "text" in task and task["text"]
            has_coords = "coordinates" in task and isinstance(task["coordinates"], (list, dict))
            if not has_text and not has_coords:
                return False, f"Task[{i}] 'click' requires 'text' or 'coordinates'"

    return True, ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/agent")
async def agent_endpoint(request: Request):
    # ── Session ID (optional header for multi-turn memory) ────────────────────
    session_id = request.headers.get("X-Session-ID", "default")

    # ── Parse request body ────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        log.warning("Malformed request body from session '%s'", session_id)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON request body"},
        )

    user_message: str = body.get("message", "").strip()
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"error": "Field 'message' is required and cannot be empty"},
        )

    log.info("[session=%s] Received: %s", session_id, user_message)

    # ── Call NVIDIA API (with retry) ──────────────────────────────────────────
    raw_response = call_nvidia_api(session_id, user_message)
    if raw_response is None:
        return JSONResponse(
            status_code=502,
            content={"error": "Failed to reach AI model after retries"},
        )

    # ── Extract JSON ──────────────────────────────────────────────────────────
    parsed = extract_json(raw_response)
    if parsed is None:
        log.error("[session=%s] JSON extraction failed: %s", session_id, raw_response)
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid AI response"},
        )

    # ── Validate tasks ────────────────────────────────────────────────────────
    valid, error_msg = validate_tasks(parsed)
    if not valid:
        log.error("[session=%s] Validation failed: %s", session_id, error_msg)
        return JSONResponse(
            status_code=422,
            content={"error": f"Invalid task structure: {error_msg}"},
        )

    # ── Persist to conversation memory ────────────────────────────────────────
    push_history(session_id, "user",      user_message)
    push_history(session_id, "assistant", raw_response)

    tasks = parsed["tasks"]
    log.info("[session=%s] Returning %d task(s): %s", session_id, len(tasks), tasks)

    return JSONResponse(content={"tasks": tasks})


@app.delete("/agent/history")
async def clear_history_endpoint(request: Request):
    """Clear conversation memory for a session."""
    session_id = request.headers.get("X-Session-ID", "default")
    clear_history(session_id)
    log.info("[session=%s] History cleared", session_id)
    return JSONResponse(content={"status": "history cleared", "session_id": session_id})


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "version":     "2.0.0",
        "model":       NVIDIA_MODEL,
        "api_key_set": bool(NVIDIA_API_KEY),
        "max_retries": MAX_RETRIES,
        "max_history": MAX_HISTORY_TURNS,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    log.info("Starting Android AI Agent Backend v2.0.0 on port %d", port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
