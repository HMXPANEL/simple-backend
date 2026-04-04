import os
import json
import re
import logging
import time
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Logging ─────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("jarvis")

# ── App ─────────────────────────────────────────
app = FastAPI(title="Jarvis Vision AGI Backend", version="7.0.0")

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

# ── Action System ───────────────────────────────
ALLOWED_ACTIONS = {
    "open_app", "click", "type", "wait", "scroll"
}

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

# ── AI CALL (VISION ENABLED) ────────────────────
def call_ai(messages):
    if not NVIDIA_API_KEY:
        return None

    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 250
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY},
        "Content-Type": "application/json"
    }

    try:
        res = http.post(NVIDIA_BASE_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"AI error: {e}")
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

# ── HEURISTICS ──────────────────────────────────
def heuristic(ui):
    ui = ui.lower()

    if "allow" in ui:
        return {"tasks":[{"action":"click","text":"allow"}]}

    if "ok" in ui:
        return {"tasks":[{"action":"click","text":"ok"}]}

    return None

# ── PLANNER ─────────────────────────────────────
PLANNER_PROMPT = """
Break goal into minimal UI steps.

Return JSON:
{"plan":[...]}

No explanation.
"""

def create_plan(goal):
    messages = [
        {"role":"system","content":PLANNER_PROMPT},
        {"role":"user","content":goal}
    ]
    raw = call_ai(messages)
    parsed = extract_json(raw)
    return parsed.get("plan", []) if parsed else []

# ── EXECUTOR (VISION ENABLED) ───────────────────
STEP_PROMPT = """
You are a vision Android agent.

Input:
- GOAL
- UI TEXT
- SCREEN IMAGE
- LAST RESULT
- FAILED ACTIONS

Output:
- ONE next action

Rules:
- Prefer text click
- If not found → use coordinates [x,y]
- Avoid repeating failures
- Be fast

Return JSON:
{"tasks":[{...}]}
"""

# ── VALIDATION ──────────────────────────────────
def validate(task):
    return task and task.get("action") in ALLOWED_ACTIONS

# ── STEP LOOP ───────────────────────────────────
@app.post("/agent/step")
async def step(request: Request):
    body = await request.json()
    sid = request.headers.get("X-Session-ID", "default")

    goal = body.get("goal", "")
    ui = body.get("ui", "")[:1000]
    image = body.get("image", "")
    last = body.get("last", "")

    session = get_session(sid)
    session["goal"] = goal

    # ── Heuristic
    h = heuristic(ui)
    if h:
        return h

    # ── Plan init
    if not session["plan"]:
        session["plan"] = create_plan(goal)

    # ── Vision + UI message
    content = [
        {"type":"text","text":f"GOAL:{goal}\nUI:{ui}\nLAST:{last}\nFAIL:{session['fail'][-3:]}"}
    ]

    if image:
        content.append({"type":"image","image":image})

    messages = [{"role":"user","content":content}]

    raw = call_ai(messages)
    parsed = extract_json(raw)

    if not parsed or not parsed.get("tasks"):
        return {"tasks":[]}

    task = parsed["tasks"][0]

    if not validate(task):
        return {"tasks":[]}

    # ── Memory update
    if last == "fail":
        session["fail"].append(task)
    else:
        session["success"].append(task)

    session["history"].append(task)

    # ── Loop breaker
    if session["history"][-3:] == [task]*3:
        return {"tasks":[{"action":"scroll","direction":"down"}]}

    return {"tasks":[task]}

# ── PLAN ROUTE ──────────────────────────────────
@app.post("/agent")
async def agent(request: Request):
    body = await request.json()
    goal = body.get("message","")

    plan = create_plan(goal)
    return {"tasks":[{"action":"type","text":p} for p in plan]}

# ── HEALTH ──────────────────────────────────────
@app.get("/health")
async def health():
    return {"status":"ok","version":"7.0.0"}

# ── RUN ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
