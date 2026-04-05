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
app = FastAPI(title="Jarvis Vision AGI Backend", version="8.0.0")

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
        "success": [],
        "plan": []
    })

# ── HTTP SESSION ────────────────────────────────
http = requests.Session()

# ── AI CALL ─────────────────────────────────────
def call_ai(messages):
    if not NVIDIA_API_KEY:
        log.error("API KEY MISSING")
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
            log.info(f"AI RAW: {content}")  # 🔥 debug
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

    if task.get("action") not in ALLOWED_ACTIONS:
        return False

    if task["action"] == "wait":
        return isinstance(task.get("duration"), int)

    if task["action"] == "scroll":
        return task.get("direction") in ["up", "down", "left", "right"]

    if task["action"] == "click":
        return "text" in task or "coordinates" in task

    return True

# ── HEURISTIC FAST BRAIN ────────────────────────
def heuristic(ui):
    ui = ui.lower()

    if "allow" in ui:
        return {"tasks":[{"action":"click","text":"allow"}]}

    if "ok" in ui:
        return {"tasks":[{"action":"click","text":"ok"}]}

    if "skip" in ui:
        return {"tasks":[{"action":"click","text":"skip"}]}

    return None

# ── STRONG PLANNER (FIXED) ──────────────────────
PLANNER_PROMPT = """
You are an Android automation planner.

Convert user request into executable tasks.

STRICT RULES:
- ONLY JSON
- NO explanation
- FORMAT:

{"tasks":[
 {"action":"open_app","app":"youtube"},
 {"action":"wait","duration":2000}
]}

Use only:
open_app, click, type, wait, scroll
"""

# ── EXECUTOR PROMPT (VISION READY) ──────────────
STEP_PROMPT = """
You are a smart Android agent.

INPUT:
GOAL: {goal}
VISIBLE UI TEXT: {ui}
LAST RESULT: {last}
FAILED: {fail}

RULES:
- Return ONE step only
- Prefer clicking visible text
- If not found → scroll
- If input needed → type
- NEVER return empty unless task done

FORMAT:
{"tasks":[{...}]}
"""

# ── PLAN ROUTE (FIXED) ──────────────────────────
@app.post("/agent")
async def agent(request: Request):
    body = await request.json()
    goal = body.get("message","")

    messages = [
        {"role":"system","content":PLANNER_PROMPT},
        {"role":"user","content":goal}
    ]

    raw = call_ai(messages)
    parsed = extract_json(raw)

    if not parsed or not parsed.get("tasks"):
        return JSONResponse(
            status_code=422,
            content={"error":"AI failed"}
        )

    return {"tasks": parsed["tasks"]}

# ── STEP LOOP (IMPROVED) ────────────────────────
@app.post("/agent/step")
async def step(request: Request):
    body = await request.json()
    sid = request.headers.get("X-Session-ID", "default")

    goal = body.get("goal","")
    ui = body.get("ui","")[:1200]
    last = body.get("last","")

    session = get_session(sid)
    session["goal"] = goal

    # FAST heuristic
    h = heuristic(ui)
    if h:
        return h

    # AI step decision
    prompt = STEP_PROMPT.format(
        goal=goal,
        ui=ui,
        last=last,
        fail=session["fail"][-3:]
    )

    messages = [{"role":"user","content":prompt}]
    raw = call_ai(messages)
    parsed = extract_json(raw)

    if not parsed or not parsed.get("tasks"):
        return {"tasks":[]}

    task = parsed["tasks"][0]

    if not validate(task):
        return {"tasks":[]}

    # memory
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
        "version":"8.0.0",
        "api_key": bool(NVIDIA_API_KEY),
        "model": NVIDIA_MODEL
    }

# ── RUN ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
