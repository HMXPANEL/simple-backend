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
app = FastAPI(title="Android AI Agent Backend", version="3.0.0")

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
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", 6))

# ── Action System ─────────────────────────────────────────────────────────────
ALLOWED_ACTIONS = {
    "open_app",
    "send_message",
    "click",
    "type",
    "wait",
    "scroll",
    "search"
}

ACTION_REQUIRED_FIELDS = {
    "open_app": ["app"],
    "send_message": ["app", "contact", "message"],
    "click": [],
    "type": ["text"],
    "wait": ["duration"],
    "scroll": ["direction"],
    "search": ["query"]
}

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Android AI agent.

Convert user input into structured JSON tasks.

STRICT RULES:
- ONLY return JSON
- NO text, NO explanation
- ALWAYS return: {"tasks": [...]}

ACTIONS:
- open_app → app
- send_message → app, contact, message
- click → text OR coordinates [x,y]
- type → text
- wait → duration (ms)
- scroll → direction (up/down/left/right)
- search → query

Return multiple tasks if needed.
"""

# ── Memory ────────────────────────────────────────────────────────────────────
_conversation_store = {}
MAX_SESSIONS = 50

def get_history(session_id):
    return _conversation_store.get(session_id, [])

def push_history(session_id, role, content):
    if len(_conversation_store) > MAX_SESSIONS:
        _conversation_store.pop(next(iter(_conversation_store)))

    history = _conversation_store.setdefault(session_id, [])
    history.append({"role": role, "content": content})

    max_msgs = MAX_HISTORY_TURNS * 2
    if len(history) > max_msgs:
        _conversation_store[session_id] = history[-max_msgs:]

def clear_history(session_id):
    _conversation_store.pop(session_id, None)

# ── Rate Limit ────────────────────────────────────────────────────────────────
LAST_REQUEST = {}

def check_rate_limit(session_id):
    now = time.time()
    if session_id in LAST_REQUEST:
        if now - LAST_REQUEST[session_id] < 1:
            return False
    LAST_REQUEST[session_id] = now
    return True

# ── AI Call ───────────────────────────────────────────────────────────────────
def build_messages(session_id, user_message):
    return [{"role": "system", "content": SYSTEM_PROMPT}] + \
           get_history(session_id) + \
           [{"role": "user", "content": user_message}]

def call_nvidia_api(session_id, user_message):
    if not NVIDIA_API_KEY:
        return None

    payload = {
        "model": NVIDIA_MODEL,
        "messages": build_messages(session_id, user_message),
        "temperature": 0.2,
        "max_tokens": 512
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            res = requests.post(
                NVIDIA_BASE_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning(f"Retry {attempt+1}: {e}")
            time.sleep(1.5 * (attempt + 1))

    return None

# ── JSON Extraction ───────────────────────────────────────────────────────────
def extract_json(raw):
    if not raw:
        return None

    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        return json.loads(raw)
    except:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None

# ── Validation ────────────────────────────────────────────────────────────────
def validate_tasks(data):
    if not isinstance(data, dict):
        return False, "Not JSON object"

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return False, "Invalid tasks"

    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            return False, f"Task[{i}] invalid"

        action = task.get("action")
        if action not in ALLOWED_ACTIONS:
            return False, f"Invalid action {action}"

        for field in ACTION_REQUIRED_FIELDS.get(action, []):
            if field not in task:
                return False, f"{action} missing {field}"

        if action == "wait":
            if not isinstance(task.get("duration"), int):
                return False, "wait duration invalid"

        if action == "scroll":
            if task.get("direction") not in ["up","down","left","right"]:
                return False, "invalid scroll direction"

        if action == "click":
            if not (task.get("text") or (
                isinstance(task.get("coordinates"), list)
                and len(task["coordinates"]) == 2
            )):
                return False, "click invalid"

    return True, ""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/agent")
async def agent(request: Request):
    session_id = request.headers.get("X-Session-ID", "default")

    if not check_rate_limit(session_id):
        return JSONResponse(status_code=429, content={"error": "Too fast"})

    try:
        body = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Bad JSON"})

    message = body.get("message", "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"error": "Empty message"})

    raw = call_nvidia_api(session_id, message)
    if not raw:
        return JSONResponse(status_code=502, content={"error": "AI failed"})

    parsed = extract_json(raw)
    if not parsed:
        return JSONResponse(status_code=422, content={"error": "Bad AI response"})

    valid, err = validate_tasks(parsed)
    if not valid:
        return JSONResponse(status_code=422, content={"error": err})

    # Save CLEAN JSON (fixed)
    push_history(session_id, "user", message)
    push_history(session_id, "assistant", json.dumps(parsed))

    return {"tasks": parsed["tasks"]}

@app.delete("/agent/history")
async def clear(request: Request):
    session_id = request.headers.get("X-Session-ID", "default")
    clear_history(session_id)
    return {"status": "cleared"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "model": NVIDIA_MODEL
    }

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
