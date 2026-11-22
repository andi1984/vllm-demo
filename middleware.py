"""Tool-calling middleware between a chat client and vLLM.

- Loads an OpenAPI spec and converts every operation into a chat-completions `tool`.
- Exposes an OpenAI-compatible POST /v1/chat/completions.
- Forwards requests to vLLM with the tools injected.
- Detects tool calls via the chat-completions protocol
  (`choices[0].finish_reason == "tool_calls"` and `message.tool_calls`),
  executes them against the real HTTP API, appends `role: "tool"` results,
  and loops until the model returns a normal answer.

Run: uvicorn middleware:app --port 8002
Env: VLLM_URL (default http://127.0.0.1:8000), TOOL_API_URL (default http://127.0.0.1:8001)
"""
import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
TOOL_API_URL = os.environ.get("TOOL_API_URL", "http://127.0.0.1:8001")
MAX_TOOL_ROUNDS = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [middleware] %(message)s")
log = logging.getLogger("middleware")

# operation_id -> {"method", "path", "params": [{"name", "in"}]}
OPERATIONS: dict[str, dict] = {}
TOOLS: list[dict] = []


def openapi_to_tools(spec: dict) -> tuple[list[dict], dict[str, dict]]:
    """Convert OpenAPI operations (query/path params) into OpenAI function tools."""
    tools, ops = [], {}
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            op_id = op.get("operationId")
            if not op_id:
                continue
            properties, required, params = {}, [], []
            for p in op.get("parameters", []):
                if p.get("in") not in ("query", "path"):
                    continue
                schema = dict(p.get("schema", {"type": "string"}))
                schema.pop("title", None)
                if "description" in p:
                    schema["description"] = p["description"]
                properties[p["name"]] = schema
                if p.get("required"):
                    required.append(p["name"])
                params.append({"name": p["name"], "in": p["in"]})
            tools.append({
                "type": "function",
                "function": {
                    "name": op_id,
                    "description": op.get("summary") or op.get("description", ""),
                    "parameters": {"type": "object", "properties": properties, "required": required},
                },
            })
            ops[op_id] = {"method": method.upper(), "path": path, "params": params}
    return tools, ops


async def execute_tool(client: httpx.AsyncClient, name: str, arguments: dict) -> str:
    op = OPERATIONS.get(name)
    if op is None:
        return json.dumps({"error": f"unknown tool {name}"})
    path, query = op["path"], {}
    for p in op["params"]:
        if p["name"] in arguments:
            if p["in"] == "path":
                path = path.replace("{" + p["name"] + "}", str(arguments[p["name"]]))
            else:
                query[p["name"]] = arguments[p["name"]]
    resp = await client.request(op["method"], TOOL_API_URL + path, params=query)
    return resp.text


@asynccontextmanager
async def lifespan(app: FastAPI):
    global TOOLS, OPERATIONS
    async with httpx.AsyncClient() as client:
        spec = (await client.get(f"{TOOL_API_URL}/openapi.json")).json()
    TOOLS, OPERATIONS = openapi_to_tools(spec)
    log.info("Loaded %d tool(s) from OpenAPI: %s", len(TOOLS), [t["function"]["name"] for t in TOOLS])
    yield


app = FastAPI(title="Tool-calling middleware", lifespan=lifespan)


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient() as client:
        return (await client.get(f"{VLLM_URL}/v1/models")).json()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    body["stream"] = False  # keep the demo simple
    body.setdefault("tools", TOOLS)
    body.setdefault("tool_choice", "auto")
    messages = list(body["messages"])
    trace = []  # tool calls executed by the middleware, returned to the client

    async with httpx.AsyncClient(timeout=300) as client:
        for _ in range(MAX_TOOL_ROUNDS):
            resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json={**body, "messages": messages})
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]

            # Chat-completions protocol signal: the model wants to call a function.
            if choice.get("finish_reason") != "tool_calls" or not msg.get("tool_calls"):
                data["x_tool_trace"] = trace
                return JSONResponse(data)

            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
            for call in msg["tool_calls"]:
                name = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                log.info(">>> LLM is calling %s(%s)  [id=%s]", name, args, call["id"])
                result = await execute_tool(client, name, args)
                log.info("<<< %s returned %s", name, result)
                trace.append({"name": name, "arguments": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    return JSONResponse({"error": "too many tool-call rounds", "x_tool_trace": trace}, status_code=500)
