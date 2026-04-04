import os
import json
import re
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Android AI Agent Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODEL   = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions",
)

SYSTEM_PROMPT = """You are an Android AI agent.
Convert every user request into structured JSON.

Rules:
- ONLY return valid JSON
- No explanations, no extra text
- Always return a 'tasks' array
- Each task must be executable on Android

Allowed actions:
- open_app (app)
- send_message (app, contact, message)
- click (text or coordinates)
- type (text)
- wait (duration in ms)

If multiple steps are needed, return multiple tasks in order."""


# ── Core functions ────────────────────────────────────────────────────────────

def call_nvidia_api(user_message: str) -> str | None:
    """Send user message to NVIDIA NIM and return raw text response."""
    if not NVIDIA_API_KEY:
        log.error("NVIDIA_API_KEY is not set")
        return None

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            NVIDIA_BASE_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        log.info("NVIDIA raw response: %s", raw)
        return raw
    except requests.exceptions.Timeout:
        log.error("NVIDIA API request timed out")
    except requests.exceptions.HTTPError as e:
        log.error("NVIDIA API HTTP error: %s", e)
    except (KeyError, IndexError, ValueError) as e:
        log.error("Unexpected NVIDIA API response structure: %s", e)
    except Exception as e:
        log.error("Unexpected error calling NVIDIA API: %s", e)

    return None


def extract_json(raw: str) -> dict | None:
    """
    Extract the first valid JSON object from a raw string.
    Handles markdown fences, leading/trailing garbage text.
    """
    if not raw:
        return None

    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt to extract the first {...} block
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    log.warning("Could not extract valid JSON from response")
    return None


def validate_tasks(data: dict) -> tuple[bool, str]:
    """
    Validate that the parsed JSON has a proper 'tasks' array
    where each task contains an 'action' key.
    Returns (is_valid, error_message).
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
            return False, f"Task at index {i} is not an object"
        if "action" not in task:
            return False, f"Task at index {i} is missing 'action' key"

    return True, ""


# ── Route ─────────────────────────────────────────────────────────────────────

@app.post("/agent")
async def agent_endpoint(request: Request):
    # ── Parse request body ────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        log.warning("Malformed request body")
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

    log.info("Received message: %s", user_message)

    # ── Call NVIDIA API ───────────────────────────────────────────────────────
    raw_response = call_nvidia_api(user_message)
    if raw_response is None:
        return JSONResponse(
            status_code=502,
            content={"error": "Failed to reach AI model"},
        )

    # ── Extract JSON ──────────────────────────────────────────────────────────
    parsed = extract_json(raw_response)
    if parsed is None:
        log.error("JSON extraction failed for response: %s", raw_response)
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid AI response"},
        )

    # ── Validate tasks ────────────────────────────────────────────────────────
    valid, error_msg = validate_tasks(parsed)
    if not valid:
        log.error("Task validation failed: %s", error_msg)
        return JSONResponse(
            status_code=422,
            content={"error": f"Invalid task structure: {error_msg}"},
        )

    log.info("Returning %d task(s)", len(parsed["tasks"]))
    return JSONResponse(content={"tasks": parsed["tasks"]})


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": NVIDIA_MODEL,
        "api_key_set": bool(NVIDIA_API_KEY),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    log.info("Starting Android AI Agent Backend on port %d", port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)