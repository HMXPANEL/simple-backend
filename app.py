import os
import json
import re
import logging
import time
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ── Logging ─────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("jarvis")

# ── App ─────────────────────────────────────────
app = FastAPI(title="Jarvis Vision AGI Backend", version="8.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ──────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions"
)

REQUEST_TIMEOUT = 20
MAX_RETRIES = 2

# ── Action System ───────────────────────────────
ALLOWED_ACTIONS = {"open_app", "click", "type", "wait", "scroll"}

# ── MEMORY ──────────────────────────────────────
SESSIONS = {}

def get_session(sid):
    return SESSIONS.setdefault(sid, {
        "goal": "",
        "history": [],
        "fail": [],
        "success": []
    })

# ── HTTP SESSION ────────────────────────────────
http = requests.Session()

# ── AI CALL ─────────────────────────────────────
def call_ai(messages):
    if not NVIDIA_API_KEY:
        log.error("❌ API KEY MISSING")
        return None

    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 300
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            res = http.post(
                NVIDIA_BASE_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            res.raise_for_status()

            content = res.json()["choices"][0]["message"]["content"]
            log.info(f"AI RAW: {content}")
            return content

        except Exception as e:
            log.warning(f"Retry {attempt+1}: {e}")
            time.sleep(1.5 * (attempt + 1))

    return None

# ── JSON EXTRACT ────────────────────────────────
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

# ── VALIDATION ──────────────────────────────────
def validate(task):
    if not task:
        return False

    action = task.get("action")

    if action not in ALLOWED_ACTIONS:
        return False

    if action == "wait":
        return isinstance(task.get("duration"), int)

    if action == "scroll":
        return task.get("direction") in ["up", "down", "left", "right"]

    if action == "click":
        return ("text" in task) or ("coordinates" in task)

    if action == "type":
        return "text" in task

    return True

# ── HEURISTIC (FAST BRAIN) ──────────────────────
def heuristic(ui):
    ui = ui.lower()

    keywords = ["allow", "ok", "skip", "continue", "yes"]

    for k in keywords:
        if k in ui:
            return {"tasks":[{"action":"click","text":k}]}

    return None

# ── STRONG PLANNER ──────────────────────────────
PLANNER_PROMPT = """
You are an Android automation agent.

Convert user request into executable UI tasks.

STRICT:
- ONLY JSON
- NO explanation
- Format: {"tasks":[...]}

Actions:
open_app, click, type, wait, scroll

Example:
User: open youtube and search cats

Output:
{"tasks":[
 {"action":"open_app","app":"youtube"},
 {"action":"wait","duration":2000},
 {"action":"click","text":"search"},
 {"action":"type","text":"cats"}
]}
"""

# ── STEP PROMPT ─────────────────────────────────
STEP_PROMPT = """
You are an Android agent.

GOAL: {goal}
UI: {ui}
LAST: {last}
FAILED: {fail}

Rules:
- ONE step only
- Prefer visible text
- If not found → scroll
- If input needed → type
- NEVER return empty

Return JSON:
{"tasks":[{...}]}
"""

# ── PLAN ROUTE ──────────────────────────────────
@app.post("/agent")
async def agent(request: Request):
    body = await request.json()
    goal = body.get("message", "")

    messages = [
        {"role":"system","content":PLANNER_PROMPT},
        {"role":"user","content":goal}
    ]

    raw = call_ai(messages)
    parsed = extract_json(raw)

    # 🔥 fallback (NEVER EMPTY)
    if not parsed or not parsed.get("tasks"):
        return {
            "tasks":[
                {"action":"open_app","app":"youtube"},
                {"action":"wait","duration":2000}
            ]
        }

    return {"tasks": parsed["tasks"]}

# ── STEP LOOP ───────────────────────────────────
@app.post("/agent/step")
async def step(request: Request):
    body = await request.json()
    sid = request.headers.get("X-Session-ID", "default")

    goal = body.get("goal", "")
    ui = body.get("ui", "")[:1200]
    last = body.get("last", "")

    session = get_session(sid)
    session["goal"] = goal

    # heuristic fast action
    h = heuristic(ui)
    if h:
        return h

    prompt = STEP_PROMPT.format(
        goal=goal,
        ui=ui,
        last=last,
        fail=session["fail"][-3:]
    )

    raw = call_ai([{"role":"user","content":prompt}])
    parsed = extract_json(raw)

    if not parsed or not parsed.get("tasks"):
        return {
            "tasks":[
                {"action":"scroll","direction":"down"}
            ]
        }

    task = parsed["tasks"][0]

    if not validate(task):
        return {
            "tasks":[
                {"action":"scroll","direction":"down"}
            ]
        }

    # memory update
    if last == "fail":
        session["fail"].append(task)
    else:
        session["success"].append(task)

    session["history"].append(task)

    # loop breaker
    if len(session["history"]) >= 3 and session["history"][-3:] == [task]*3:
        return {"tasks":[{"action":"scroll","direction":"down"}]}

    return {"tasks":[task]}

# ── HEALTH ──────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":"ok",
        "version":"8.1.0",
        "api_key": bool(NVIDIA_API_KEY),
        "model": NVIDIA_MODEL
    }

# ── RUN ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
