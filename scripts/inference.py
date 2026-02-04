import litellm
import time
import json
API_BASE = "https://hwu70zhcee9cmp-8000.proxy.runpod.net/v1"  # Update this

API_KEY = "sk-123456"
MODEL = "openai/Qwen/Qwen2.5-3B"
prompts = ["who has written Harry Potter ? ", "whats the weather in sanfrancisco ?"]
TOOLS = []
for prompt in prompts:
    start = time.time()
    response = litellm.completion(
            model=MODEL, api_base=API_BASE, api_key=API_KEY,
            messages=[{"role": "user", "content": prompt}],
            tools=TOOLS, tool_choice="auto"
        )
    elapsed = time.time() - start
    msg = response.choices[0].message.content
    tokens = response["usage"]["completion_tokens"]
    print(f"Generated: {tokens} tokens in {elapsed:.2f}s")
    print(f"Speed: {tokens/elapsed:.1f} tokens/sec")
    print(msg)