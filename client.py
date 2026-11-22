"""Chat client using the standard OpenAI SDK against the middleware.

Run: python client.py "Please say hello to Ada"
"""
import sys

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="not-needed")

prompt = " ".join(sys.argv[1:]) or "Use the hello_world tool to greet Ada."
model = client.models.list().data[0].id

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)

trace = (resp.model_extra or {}).get("x_tool_trace", [])
print(f"User:  {prompt}")
if trace:
    for t in trace:
        print(f"Tool:  LLM called {t['name']}({t['arguments']}) -> {t['result']}")
else:
    print("Tool:  (no tool call - model answered directly)")
print(f"LLM:   {resp.choices[0].message.content}")
