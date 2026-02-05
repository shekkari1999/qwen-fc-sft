"""
TARS - Function Calling Assistant
Gradio UI with vLLM backend serving multiple LoRA adapters
"""
import gradio as gr
import requests
import json
import time
from typing import Generator

# vLLM server config (RunPod endpoint)
VLLM_URL = "https://tch5lmp6hj0b9h-8000.proxy.runpod.net/v1/chat/completions"

# Available tools for the model
AVAILABLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {"location": {"type": "str", "description": "City name"}}
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {"expression": {"type": "str", "description": "Math expression"}}
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {"query": {"type": "str", "description": "Search query"}}
    }
]

SYSTEM_PROMPT = """You are TARS, a helpful assistant with access to the following tools:

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

STAGE_INFO = {
    "stage1": {
        "name": "Stage 1: Chat",
        "model": "stage1",
        "description": "Base chat capability - SFT on conversation data",
        "system": "You are a helpful assistant."
    },
    "stage2": {
        "name": "Stage 2: Function Calling",
        "model": "stage2",
        "description": "Learned <tool_call> format - SFT on function calling data",
        "system": SYSTEM_PROMPT
    },
    "tars": {
        "name": "Stage 3: TARS (Final)",
        "model": "tars",
        "description": "Knows WHEN to use tools - DPO trained on preferences",
        "system": SYSTEM_PROMPT
    }
}


def format_tools_display():
    """Format tools for sidebar display"""
    tools_md = "## Available Tools\n\n"
    for tool in AVAILABLE_TOOLS:
        tools_md += f"### `{tool['name']}`\n"
        tools_md += f"{tool['description']}\n\n"
        tools_md += "**Parameters:**\n"
        for param, info in tool['parameters'].items():
            tools_md += f"- `{param}`: {info.get('description', info['type'])}\n"
        tools_md += "\n---\n\n"
    return tools_md


def chat_with_model(
    message: str,
    history: list,
    stage: str,
    temperature: float,
    max_tokens: int,
    top_p: float
) -> Generator:
    """Stream chat responses from vLLM"""

    stage_config = STAGE_INFO[stage]

    # Build messages
    messages = [{"role": "system", "content": stage_config["system"]}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    # Request payload
    payload = {
        "model": stage_config["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": True
    }

    start_time = time.time()
    tokens_generated = 0
    response_text = ""

    try:
        with requests.post(VLLM_URL, json=payload, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                yield f"Error: {resp.status_code} - {resp.text}", ""
                return

            for line in resp.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if chunk["choices"][0]["delta"].get("content"):
                                token = chunk["choices"][0]["delta"]["content"]
                                response_text += token
                                tokens_generated += 1

                                # Calculate stats
                                elapsed = time.time() - start_time
                                tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

                                stats = f"""**Tokens:** {tokens_generated} | **Time:** {elapsed:.2f}s | **Speed:** {tokens_per_sec:.1f} tok/s | **Model:** {stage_config['model']}"""

                                yield response_text, stats
                        except json.JSONDecodeError:
                            continue

    except requests.exceptions.ConnectionError:
        yield "**Error:** Cannot connect to vLLM server. Make sure it's running on localhost:8000", ""
    except Exception as e:
        yield f"**Error:** {str(e)}", ""

    # Final stats
    elapsed = time.time() - start_time
    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

    # Check if tool was called
    tool_used = "Yes" if "<tool_call>" in response_text else "No"

    final_stats = f"""**Tokens:** {tokens_generated} | **Time:** {elapsed:.2f}s | **Speed:** {tokens_per_sec:.1f} tok/s | **Model:** {stage_config['model']} | **Tool Called:** {tool_used}"""

    yield response_text, final_stats


def create_ui():
    """Create the Gradio interface"""

    # Custom CSS for classy dark theme
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .stats-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 12px;
        color: #e94560;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85em;
    }
    .tool-sidebar {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 16px;
    }
    .stage-radio label {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 4px;
        transition: all 0.2s;
    }
    .header-title {
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 700;
    }
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Base(primary_hue="red", neutral_hue="slate")) as demo:

        # Header
        gr.HTML("""
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="font-size: 2.5em; margin: 0; background: linear-gradient(90deg, #e94560, #4a90d9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    TARS
                </h1>
                <p style="color: #888; margin-top: 8px; font-size: 1.1em;">
                    Function Calling Assistant | Honesty 90% | Tool Use 100%
                </p>
            </div>
        """)

        with gr.Row():
            # Left sidebar - Tools
            with gr.Column(scale=1, min_width=250):
                gr.HTML("""<h3 style="color: #e94560; margin-bottom: 16px;">🛠️ Tools</h3>""")

                for tool in AVAILABLE_TOOLS:
                    with gr.Group():
                        gr.HTML(f"""
                            <div style="padding: 12px; background: #1a1a2e; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #e94560;">
                                <code style="color: #4a90d9; font-size: 1.1em;">{tool['name']}</code>
                                <p style="color: #aaa; font-size: 0.9em; margin: 8px 0 0 0;">{tool['description']}</p>
                            </div>
                        """)

                gr.HTML("""<hr style="border-color: #333; margin: 20px 0;">""")
                gr.HTML("""<h3 style="color: #e94560; margin-bottom: 16px;">⚙️ Model Stage</h3>""")

                stage_select = gr.Radio(
                    choices=[
                        ("Stage 1: Chat", "stage1"),
                        ("Stage 2: Function Calling", "stage2"),
                        ("Stage 3: TARS (Final)", "tars")
                    ],
                    value="tars",
                    label="",
                    interactive=True
                )

                stage_desc = gr.Markdown(
                    value=STAGE_INFO["tars"]["description"],
                    elem_classes=["stage-description"]
                )

                def update_stage_desc(stage):
                    return STAGE_INFO[stage]["description"]

                stage_select.change(update_stage_desc, stage_select, stage_desc)

            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=tars"),
                    elem_classes=["chat-window"]
                )

                # Stats display
                stats_display = gr.Markdown(
                    value="**Ready** | Select a model stage and start chatting",
                    elem_classes=["stats-box"]
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask TARS anything... (try: 'What's the weather in Tokyo?' or 'Hello!')",
                        label="",
                        scale=6,
                        container=False
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", size="sm")

            # Right sidebar - Parameters
            with gr.Column(scale=1, min_width=200):
                gr.HTML("""<h3 style="color: #e94560; margin-bottom: 16px;">🎛️ Parameters</h3>""")

                temperature = gr.Slider(
                    minimum=0, maximum=1.5, value=0.1, step=0.05,
                    label="Temperature",
                    info="Lower = more deterministic"
                )

                max_tokens = gr.Slider(
                    minimum=50, maximum=1024, value=256, step=32,
                    label="Max Tokens",
                    info="Maximum response length"
                )

                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                    label="Top P",
                    info="Nucleus sampling threshold"
                )

                gr.HTML("""<hr style="border-color: #333; margin: 20px 0;">""")
                gr.HTML("""<h3 style="color: #e94560; margin-bottom: 16px;">📊 Legend</h3>""")
                gr.HTML("""
                    <div style="font-size: 0.85em; color: #888;">
                        <p><strong>Tokens:</strong> Output tokens generated</p>
                        <p><strong>Speed:</strong> Tokens per second</p>
                        <p><strong>Tool Called:</strong> Whether &lt;tool_call&gt; was used</p>
                    </div>
                """)

        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666; font-size: 0.9em;">
                <p>Built with Qwen2.5-3B + LoRA | 3-Stage Training Pipeline | vLLM Serving</p>
                <p style="margin-top: 4px;">Stage 1: SFT Chat → Stage 2: SFT Function Calling → Stage 3: DPO Tool Decisions</p>
            </div>
        """)

        # Event handlers
        def user_message(message, history):
            return "", history + [[message, None]]

        def bot_response(history, stage, temperature, max_tokens, top_p):
            message = history[-1][0]
            history_prev = history[:-1]

            for response, stats in chat_with_model(message, history_prev, stage, temperature, max_tokens, top_p):
                history[-1][1] = response
                yield history, stats

        # Submit on enter or button click
        msg_input.submit(
            user_message, [msg_input, chatbot], [msg_input, chatbot]
        ).then(
            bot_response, [chatbot, stage_select, temperature, max_tokens, top_p], [chatbot, stats_display]
        )

        send_btn.click(
            user_message, [msg_input, chatbot], [msg_input, chatbot]
        ).then(
            bot_response, [chatbot, stage_select, temperature, max_tokens, top_p], [chatbot, stats_display]
        )

        clear_btn.click(lambda: ([], "**Ready** | Cleared"), None, [chatbot, stats_display])

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
