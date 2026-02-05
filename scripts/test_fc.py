"""
Test function calling capabilities (before and after Stage 2 training)
Usage: python scripts/test_fc.py --model ./checkpoints/stage1/final
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import argparse
import json

# Sample tools for testing
SAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "location": {"type": "str", "description": "City name, e.g., 'Paris'"},
            "unit": {"type": "str", "description": "Temperature unit: 'celsius' or 'fahrenheit'", "default": "celsius"}
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "str", "description": "Search query"}
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "expression": {"type": "str", "description": "Math expression to evaluate, e.g., '2 + 2'"}
        }
    }
]

# Test cases
TEST_CASES = [
    "What's the weather in Tokyo?",
    "Search for the latest news about AI",
    "Calculate 15 * 7 + 23",
    "What's the weather in London in fahrenheit?",
    "I need to know the weather in Paris and also search for French restaurants",
]

SYSTEM_PROMPT = """You are a helpful assistant with access to the following tools:

{tools}

When you need to use a tool, respond with:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>

If you don't need a tool, just respond normally."""


def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def test_function_calling(model, tokenizer, test_cases=TEST_CASES):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    system_msg = SYSTEM_PROMPT.format(tools=json.dumps(SAMPLE_TOOLS, indent=2))

    print("\n" + "="*70)
    print("FUNCTION CALLING TEST")
    print("="*70)
    print(f"\nSystem prompt includes {len(SAMPLE_TOOLS)} tools: {[t['name'] for t in SAMPLE_TOOLS]}")
    print("\nExpected output format:")
    print('<tool_call>')
    print('{"name": "tool_name", "arguments": {"arg": "value"}}')
    print('</tool_call>')
    print("="*70)

    results = {"passed": 0, "failed": 0, "details": []}

    for i, user_query in enumerate(test_cases, 1):
        print(f"\n--- Test {i}/{len(test_cases)} ---")
        print(f"User: {user_query}")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_query}
        ]

        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end_id,
            )

        generated = outputs[0][inputs.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        stopped = im_end_id in generated.tolist()

        # Check if response contains tool_call
        has_tool_call = "<tool_call>" in response and "</tool_call>" in response

        # Try to parse the tool call
        valid_json = False
        if has_tool_call:
            try:
                start = response.index("<tool_call>") + len("<tool_call>")
                end = response.index("</tool_call>")
                tool_json = response[start:end].strip()
                parsed = json.loads(tool_json)
                if "name" in parsed and "arguments" in parsed:
                    valid_json = True
            except:
                pass

        # Determine result
        if has_tool_call and valid_json and stopped:
            status = "✅ PASS"
            results["passed"] += 1
        elif has_tool_call and valid_json:
            status = "⚠️ PARTIAL (didn't stop)"
            results["failed"] += 1
        elif has_tool_call:
            status = "⚠️ PARTIAL (invalid JSON)"
            results["failed"] += 1
        else:
            status = "❌ FAIL (no tool_call)"
            results["failed"] += 1

        print(f"AI: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"[stopped: {'YES' if stopped else 'NO'}, has_tool_call: {has_tool_call}, valid_json: {valid_json}]")
        print(f"Result: {status}")

        results["details"].append({
            "query": user_query,
            "response": response,
            "stopped": stopped,
            "has_tool_call": has_tool_call,
            "valid_json": valid_json,
            "status": status
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Passed: {results['passed']}/{len(test_cases)}")
    print(f"Failed: {results['failed']}/{len(test_cases)}")
    print(f"Success Rate: {results['passed']/len(test_cases)*100:.1f}%")

    return results


def interactive_mode(model, tokenizer):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    system_msg = SYSTEM_PROMPT.format(tools=json.dumps(SAMPLE_TOOLS, indent=2))

    print("\n" + "="*70)
    print("INTERACTIVE FUNCTION CALLING TEST")
    print(f"Tools available: {[t['name'] for t in SAMPLE_TOOLS]}")
    print("Type 'quit' to exit")
    print("="*70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_input}
            ]

            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=im_end_id,
                )

            generated = outputs[0][inputs.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            stopped = im_end_id in generated.tolist()
            has_tool_call = "<tool_call>" in response

            print(f"AI: {response}")
            print(f"   [stopped: {'YES' if stopped else 'NO'}, tool_call: {'YES' if has_tool_call else 'NO'}]\n")

        except KeyboardInterrupt:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/stage1/final")
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        test_function_calling(model, tokenizer)


if __name__ == "__main__":
    main()
