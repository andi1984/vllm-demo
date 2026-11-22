"""Hello-world tool service. FastAPI auto-publishes its OpenAPI spec at /openapi.json.

Run: uvicorn hello_api:app --port 8001
"""
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="Hello API", version="1.0.0")


class Greeting(BaseModel):
    message: str


@app.get(
    "/hello",
    operation_id="hello_world",  # becomes the tool/function name
    summary="Greet a person by name and return a hello-world message.",
    response_model=Greeting,
)
def hello_world(name: str = Query("World", description="Name of the person to greet")) -> Greeting:
    return Greeting(message=f"Hello, {name}! 👋 (from hello_api)")
