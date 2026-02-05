"""
Test Stage 3 model - checks both tool use AND no-tool scenarios
Usage: python scripts/test_stage3.py --model ./checkpoints/stage3/final
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import argparse

SYSTEM_PROMPT = """You are a helpful assistant with access to the following tools:

[
  {"name": "get_weather", "description": "Get current weather for a location", "parameters": {"location": {"type": "str"}}},
  {"name": "search_web", "description": "Search the web for information", "parameters": {"query": {"type": "str"}}},
  {"name": "calculate", "description": "Perform mathematical calculations", "parameters": {"expression": {"type": "str"}}}
]

When you need to use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

Only use tools when necessary. For greetings, general knowledge, or casual conversation, respond directly."""

# Test cases: (question, should_use_tool, expected_tool_or_None)
TEST_CASES = [
    # SHOULD use tools
    ("What's the weather in Paris?", True, "get_weather"),
    ("Is it raining in London right now?", True, "get_weather"),
    ("Calculate 456 * 789", True, "calculate"),
    ("What's 15% of 2500?", True, "calculate"),
    ("Search for the latest iPhone reviews", True, "search_web"),
    ("What are the top news headlines today?", True, "search_web"),

    # Should NOT use tools - greetings
    ("Hello!", False, None),
    ("Hi, how are you?", False, None),
    ("Good morning!", False, None),
    ("Thanks for your help!", False, None),

    # Should NOT use tools - general knowledge
    ("What is the capital of Japan?", False, None),
    ("Who wrote Romeo and Juliet?", False, None),
    ("What is photosynthesis?", False, None),
    ("How does gravity work?", False, None),
    ("Why is the sky blue?", False, None),

    # Edge cases - conceptual vs real-time
    ("What causes weather?", False, None),  # Conceptual - NO tool
    ("How do calculators work?", False, None),  # Conceptual - NO tool
    ("What's 2 + 2?", False, None),  # Simple math - NO tool
]


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


def test_model(model, tokenizer):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    results = {"correct": 0, "wrong": 0, "details": []}

    print("\n" + "="*70)
    print("STAGE 3 TEST: Tool Use Decisions")
    print("="*70)

    for question, should_use_tool, expected_tool in TEST_CASES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end_id,
            )

        generated = outputs[0][inputs.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Check if tool was used
        used_tool = "<tool_call>" in response

        # Determine correctness
        if should_use_tool and used_tool:
            status = "✅ CORRECT"
            results["correct"] += 1
        elif not should_use_tool and not used_tool:
            status = "✅ CORRECT"
            results["correct"] += 1
        elif should_use_tool and not used_tool:
            status = "❌ WRONG (should have used tool)"
            results["wrong"] += 1
        else:  # not should_use_tool and used_tool
            status = "❌ WRONG (shouldn't use tool)"
            results["wrong"] += 1

        # Print result
        expected_str = f"USE {expected_tool}" if should_use_tool else "NO tool"
        print(f"\nQ: {question}")
        print(f"Expected: {expected_str}")
        print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        print(f"Result: {status}")

        results["details"].append({
            "question": question,
            "expected_tool": should_use_tool,
            "used_tool": used_tool,
            "correct": (should_use_tool == used_tool),
            "response": response
        })

    # Summary
    total = results["correct"] + results["wrong"]
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Correct: {results['correct']}/{total} ({100*results['correct']/total:.1f}%)")
    print(f"Wrong: {results['wrong']}/{total}")

    # Breakdown
    tool_cases = [d for d in results["details"] if d["expected_tool"]]
    no_tool_cases = [d for d in results["details"] if not d["expected_tool"]]

    tool_correct = sum(1 for d in tool_cases if d["correct"])
    no_tool_correct = sum(1 for d in no_tool_cases if d["correct"])

    print(f"\nShould USE tool: {tool_correct}/{len(tool_cases)} correct")
    print(f"Should NOT use tool: {no_tool_correct}/{len(no_tool_cases)} correct")

    return results


def interactive_mode(model, tokenizer):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    print("\n" + "="*70)
    print("INTERACTIVE MODE - Test any question")
    print("Type 'quit' to exit")
    print("="*70 + "\n")

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]

            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=im_end_id,
                )

            generated = outputs[0][inputs.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            used_tool = "<tool_call>" in response

            print(f"AI: {response}")
            print(f"   [tool_call: {'YES' if used_tool else 'NO'}]\n")

        except KeyboardInterrupt:
            break

    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/stage3/final")
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        test_model(model, tokenizer)


if __name__ == "__main__":
    main()
