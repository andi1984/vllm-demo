"""One Swagger UI page for all OpenAPI specs in the demo.

- GET /                      Swagger UI with a dropdown to switch between services
- GET /specs/{name}.json     the service's OpenAPI spec, fetched server-side,
                             with `servers` rewritten to /proxy/{name}
- ANY /proxy/{name}/{path}   forwards "Try it out" requests to the service

Everything the browser loads comes from this one origin, so no CORS setup is
needed in the services themselves.

Run: uvicorn docs_portal:app --port 8003   then open http://127.0.0.1:8003
"""
import json

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

SERVICES = {
    "middleware": "http://127.0.0.1:8002",
    "hello_api": "http://127.0.0.1:8001",
    "vllm": "http://127.0.0.1:8000",
}

# Headers that must not be copied between connections.
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer",
              "upgrade", "proxy-authorization", "proxy-authenticate", "host",
              "content-length", "content-encoding"}

app = FastAPI(title="API docs portal", docs_url=None, redoc_url=None, openapi_url=None)


def backend(name: str) -> str:
    if name not in SERVICES:
        raise HTTPException(404, f"unknown service {name!r}")
    return SERVICES[name]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    urls = [{"name": f"{name}  ({base})", "url": f"/specs/{name}.json"} for name, base in SERVICES.items()]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>API docs portal</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
  <script>
    window.ui = SwaggerUIBundle({{
      urls: {json.dumps(urls)},
      "urls.primaryName": {json.dumps(urls[0]["name"])},
      dom_id: "#swagger-ui",
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
      layout: "StandaloneLayout",
      deepLinking: true,
    }});
  </script>
</body>
</html>"""


@app.get("/specs/{name}.json")
async def spec(name: str):
    base = backend(name)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base}/openapi.json")
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"{name} not reachable at {base}: {exc}"}, status_code=503)
    data = resp.json()
    data["servers"] = [{"url": f"/proxy/{name}", "description": f"{name} via portal proxy ({base})"}]
    return data


@app.api_route("/proxy/{name}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(name: str, path: str, request: Request):
    base = backend(name)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.request(
                request.method, f"{base}/{path}",
                params=request.query_params, headers=headers, content=await request.body(),
            )
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"{name} not reachable at {base}: {exc}"}, status_code=502)
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
    return Response(resp.content, status_code=resp.status_code, headers=out_headers)
