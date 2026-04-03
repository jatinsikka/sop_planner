import os
from typing import Final

import requests


PROMPT: Final = [
  {"role": "system", "content": "You are a concise poet."},
  {"role": "user", "content": "write a haiku about ai"},
]


def main() -> None:
  host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
  model = os.getenv("OLLAMA_MODEL", "gemma3:4b")

  # Build single prompt from messages
  prompt = "\n".join([f"{m['role']}: {m['content']}" for m in PROMPT])

  resp = requests.post(
    f"{host}/api/generate",
    json={"model": model, "prompt": prompt, "stream": False},
    timeout=120,
  )
  resp.raise_for_status()
  data = resp.json()
  print(data.get("response", ""))


if __name__ == "__main__":
  main()

