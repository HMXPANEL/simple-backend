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
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Jarvis Android AI Backend", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions",
)

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 2))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 30))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", 6))

# ── Action System ─────────────────────────────────────────────────────────────
ALLOWED_ACTIONS = {
    "open_app",
    "send_message",
    "click",
    "type",
    "wait",
    "scroll"
}

ACTION_REQUIRED_FIELDS = {
    "open_app": ["app"],
    "send_message": ["app", "contact", "message"],
    "click": [],
    "type": ["text"],
    "wait": ["duration"],
    "scroll": ["direction"]
}

# Optional smart fields
OPTIONAL_FIELDS = ["wait_for", "retry", "timeout"]

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────

# MULTI-TASK (one shot plan)
SYSTEM_PROMPT = """You are an Android AI agent.

Convert user input into LOW-LEVEL executable tasks.

Rules:
- ONLY JSON
- No explanation
- Always return {"tasks":[...]}

Do NOT use abstract actions like search.
Break into real UI steps.

Each task can include:
- action
- parameters
- wait_for
- retry
- timeout

Return multiple steps.
"""

# AUTONOMOUS STEP (loop)
STEP_PROMPT = """You are an autonomous Android agent.

Input:
- GOAL
- Current UI
- Last result

Output:
- ONE next action only

Rules:
- JSON only
- {"tasks":[{...}]}

If goal complete:
return {"tasks":[]}
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

# ── AI CALL ───────────────────────────────────────────────────────────────────
def call_ai(messages):
    if not NVIDIA_API_KEY:
        return None

    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
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

# ── JSON EXTRACTION ───────────────────────────────────────────────────────────
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

# ── VALIDATION ────────────────────────────────────────────────────────────────
def validate_tasks(data):
    if not isinstance(data, dict):
        return False, "Not JSON"

    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return False, "Invalid tasks"

    for task in tasks:
        action = task.get("action")
        if action not in ALLOWED_ACTIONS:
            return False, f"Invalid action {action}"

        for field in ACTION_REQUIRED_FIELDS.get(action, []):
            if field not in task:
                return False, f"{action} missing {field}"

        if "retry" in task and not isinstance(task["retry"], int):
            return False, "retry invalid"

        if "timeout" in task and not isinstance(task["timeout"], int):
            return False, "timeout invalid"

    return True, ""

# ── ROUTES ────────────────────────────────────────────────────────────────────

# FULL MULTI-TASK PLAN
@app.post("/agent")
async def agent(request: Request):
    session_id = request.headers.get("X-Session-ID", "default")

    if not check_rate_limit(session_id):
        return JSONResponse(status_code=429, content={"error": "Too fast"})

    body = await request.json()
    message = body.get("message", "").strip()

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}]

    raw = call_ai(messages)
    parsed = extract_json(raw)

    valid, err = validate_tasks(parsed)
    if not valid:
        return {"error": err}

    return {"tasks": parsed["tasks"]}


# AUTONOMOUS STEP LOOP
@app.post("/agent/step")
async def agent_step(request: Request):
    body = await request.json()

    goal = body.get("goal", "")
    ui = body.get("ui", "")
    last = body.get("last", "")

    messages = [
        {"role": "system", "content": STEP_PROMPT},
        {"role": "user", "content": f"GOAL:{goal}\nUI:{ui}\nLAST:{last}"}
    ]

    raw = call_ai(messages)
    parsed = extract_json(raw)

    return parsed or {"tasks": []}


@app.delete("/agent/history")
async def clear(request: Request):
    session_id = request.headers.get("X-Session-ID", "default")
    clear_history(session_id)
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "5.0.0",
        "model": NVIDIA_MODEL
    }


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
