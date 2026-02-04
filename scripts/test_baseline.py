import litellm
import json
API_BASE = "https://hwu70zhcee9cmp-8000.proxy.runpod.net/v1"  # Update this

API_KEY = "sk-123456"
MODEL = "openai/Qwen/Qwen2.5-3B"

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type":
"object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
"required": ["city"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the internet", "parameters": {"type":
"object", "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email to someone", "parameters": {"type":
"object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required":
["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "get_stock_price", "description": "Get current stock price", "parameters":
{"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Translate text to another language",
"parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required":
["text", "target_language"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Perform math calculation", "parameters": {"type":
"object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "book_flight", "description": "Book a flight", "parameters": {"type": "object",
"properties": {"from_city": {"type": "string"}, "to_city": {"type": "string"}, "date": {"type": "string"}}, "required":
["from_city", "to_city", "date"]}}},
]

# (prompt, expected_function, expected_args_subset)
TEST_CASES = [
    ("What's the weather in Tokyo?", "get_weather", {"city": "Tokyo"}),
    ("Weather in Paris in fahrenheit", "get_weather", {"city": "Paris", "unit": "fahrenheit"}),
    ("Search for Python tutorials", "search_web", {"query": "Python tutorials"}),
    ("What's Apple stock price?", "get_stock_price", {"symbol": "AAPL"}),
    ("TSLA price", "get_stock_price", {"symbol": "TSLA"}),
    ("Translate 'hello' to Spanish", "translate_text", {"text": "hello", "target_language": "Spanish"}),
    ("What's 25 * 4?", "calculate", {"expression": "25 * 4"}),
    ("Book flight from NYC to LA on March 15", "book_flight", {"from_city": "NYC", "to_city": "LA"}),
    ("My friend John who lives in Berlin mentioned it's cold there. Can you check?",  "get_weather", {"city": "Berlin"})]

def check_args(actual_args: dict, expected_subset: dict) -> tuple[bool, list]:
    """Check if expected args are present (partial match)"""
    missing = []
    wrong = []

    for key, expected_val in expected_subset.items():
        if key not in actual_args:
            missing.append(key)
        elif expected_val.lower() not in actual_args[key].lower():
            wrong.append(f"{key}: got '{actual_args[key]}' expected '{expected_val}'")

    return len(missing) == 0 and len(wrong) == 0, missing + wrong

print(f"=== BASELINE ({len(TOOLS)} tools, {len(TEST_CASES)} cases) ===\n")

func_correct = 0
args_correct = 0

for prompt, expected_func, expected_args in TEST_CASES:
    try:
        response = litellm.completion(
            model=MODEL, api_base=API_BASE, api_key=API_KEY,
            messages=[{"role": "user", "content": prompt}],
            tools=TOOLS, tool_choice="auto"
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            called_func = msg.tool_calls[0].function.name
            called_args = json.loads(msg.tool_calls[0].function.arguments)
        else:
            called_func = "none"
            called_args = {}

    except Exception as e:
        called_func = "error"
        called_args = {}

    func_ok = called_func == expected_func
    args_ok, issues = check_args(called_args, expected_args) if func_ok else (False, [])

    if func_ok:
        func_correct += 1
    if func_ok and args_ok:
        args_correct += 1

    status = "✅" if (func_ok and args_ok) else "⚠️ " if func_ok else "❌"
    print(f"{status} '{prompt[:40]}'")
    print(f"   Function: {called_func} {'✓' if func_ok else '✗ expected: ' + expected_func}")
    if func_ok:
        print(f"   Args: {called_args}")
        if issues:
            print(f"   Issues: {issues}")
    print()

print(f"=== RESULTS ===")
print(f"Function Selection: {func_correct}/{len(TEST_CASES)} ({100*func_correct/len(TEST_CASES):.0f}%)")
print(f"Full Accuracy (func + args): {args_correct}/{len(TEST_CASES)} ({100*args_correct/len(TEST_CASES):.0f}%)")

