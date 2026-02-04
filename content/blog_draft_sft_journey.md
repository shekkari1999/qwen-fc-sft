# Fine-Tuning Qwen2.5-3B for Function Calling: A Complete Journey

*From base model that outputs garbage to a production-ready function-calling assistant*

---

## Table of Contents
1. [The Goal](#the-goal)
2. [Part 1: Setting Up Inference with vLLM](#part-1-vllm-inference)
3. [Part 2: Baseline Testing](#part-2-baseline-testing)
4. [Part 3: Understanding the Data Formats](#part-3-data-formats)
5. [Part 4: Two-Stage SFT Approach](#part-4-two-stage-sft)
6. [Part 5: Training](#part-5-training) *(coming soon)*
7. [Part 6: Evaluation & Results](#part-6-results) *(coming soon)*

---

## The Goal

**Objective:** Take a base language model and fine-tune it to:
1. ✅ Follow instructions and chat naturally
2. ✅ Call functions/tools accurately when needed
3. ✅ Know when NOT to call a function

**Model Choice:** `Qwen/Qwen2.5-3B` (base model, not Instruct)

**Why base model?**
- Instruct models already do function calling well (boring!)
- Starting from base shows the full SFT pipeline
- Better learning experience
- Dramatic before/after improvement for the blog

---

## Part 1: Setting Up Inference with vLLM {#part-1-vllm-inference}

### What is vLLM?

vLLM is a high-performance inference engine for LLMs. It's not a model—it's the software that loads models and serves them efficiently.

### Key vLLM Optimizations

| Optimization | What it does |
|--------------|--------------|
| **PagedAttention** | Virtual memory for KV cache, no fragmentation |
| **Continuous Batching** | Process multiple requests simultaneously |
| **KV Cache Optimization** | Reuse computed attention states |

### Performance: Plain PyTorch vs vLLM

```
Plain PyTorch:  ~20-30 tokens/sec (sequential)
vLLM (1 req):   ~50 tokens/sec
vLLM (8 req):   ~368 tokens/sec (continuous batching!)
```

**7.5x throughput improvement** with concurrent requests!

### RunPod Deployment

We deployed on RunPod using:
- **GPU:** RTX A6000 (48GB) / RTX 4000 Ada (16GB)
- **Image:** `vllm/vllm-openai:latest`
- **Model:** `Qwen/Qwen2.5-3B`

**vLLM Environment Variables:**
```
VLLM_MODEL_NAME=Qwen/Qwen2.5-3B
VLLM_HOST=0.0.0.0
VLLM_PORT=8000
VLLM_API_KEY=sk-123456
VLLM_GPU_UTIL=0.90
VLLM_MAX_MODEL_LEN=8192
```

### VRAM Calculation

```
VRAM = Parameters × Bytes per Param

Qwen2.5-3B in BF16:
= 3,000,000,000 × 2 bytes
= 6 GB (weights only)

Total with KV cache + overhead:
≈ 8-10 GB for inference
```

---

## Part 2: Baseline Testing {#part-2-baseline-testing}

### Testing Function Calling Capability

We tested with 7 tools and various prompts:

```python
TOOLS = [
    {"name": "get_weather", ...},
    {"name": "search_web", ...},
    {"name": "send_email", ...},
    {"name": "get_stock_price", ...},
    {"name": "translate_text", ...},
    {"name": "calculate", ...},
    {"name": "book_flight", ...},
]
```

### Results: Instruct vs Base Model

| Model | Function Calling Accuracy |
|-------|---------------------------|
| Qwen2.5-7B-Instruct | 100% (9/9) |
| Qwen2.5-3B-Instruct | ~90%+ |
| Qwen2.5-3B (base) | **0%** - outputs garbage |

### Base Model Output (Before SFT)

When asked "What is the weather in Tokyo?" with tools provided:

```json
{
  "content": "ceryond<tbody>\n<head>\n<title>今日头条 (Toutiao)</title></head>...",
  "tool_calls": []  // Empty! No function called
}
```

The base model has no idea what to do with tools. It outputs random HTML/text.

**This is our starting point. After SFT, we expect 90%+ accuracy.**

---

## Part 3: Understanding the Data Formats {#part-3-data-formats}

### Target Format: Qwen Chat Template

All training data must be in this format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

### Function Calling Format

Qwen uses `<tool_call>` tags:

**Single tool call:**
```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
```

**Multiple tool calls:**
```
<tool_call>
{"name": "get_id", "arguments": {"q": "Pulp Fiction"}}
</tool_call>
<tool_call>
{"name": "search_torrents", "arguments": {"keywords": "Pulp Fiction"}}
</tool_call>
```

### Dataset 1: UltraChat (Chat/Instructions)

**Source:** `HuggingFaceH4/ultrachat_200k`
**Examples:** 50,000 (subset)
**Purpose:** Teach the model to follow instructions and chat

**Original format:**
```json
{
  "prompt": "Create a thank you card...",
  "messages": [
    {"role": "user", "content": "Create a thank you card..."},
    {"role": "assistant", "content": "[Your Name]..."},
    ...
  ]
}
```

**Conversion:** Minimal - just add system prompt

### Dataset 2: Salesforce xlam (Function Calling)

**Source:** `Salesforce/xlam-function-calling-60k`
**Examples:** 60,000
**Purpose:** Teach the model to call tools

**Original format:**
```json
{
  "query": "Identify the ID of 'Pulp Fiction' and search for torrents",
  "tools": "[{\"name\": \"get_id\", ...}]",  // JSON string!
  "answers": "[{\"name\": \"get_id\", \"arguments\": {...}}]"  // JSON string!
}
```

**Conversion:** Complex restructuring needed

```
Original                          Converted
────────                          ─────────
query           →                 user message
tools (parsed)  →                 system prompt with tool definitions
answers (parsed) →                assistant response with <tool_call> tags
```

**Converted format:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to tools:\n[...]\n\nUse <tool_call> tags when calling tools."
    },
    {
      "role": "user",
      "content": "Identify the ID of 'Pulp Fiction' and search for torrents"
    },
    {
      "role": "assistant",
      "content": "<tool_call>\n{\"name\": \"get_id\", \"arguments\": {\"q\": \"Pulp Fiction\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"search_torrents\", \"arguments\": {...}}\n</tool_call>"
    }
  ]
}
```

---

## Part 4: Two-Stage SFT Approach {#part-4-two-stage-sft}

### Why Two Stages?

| Approach | Pros | Cons |
|----------|------|------|
| Single stage (mixed data) | Simpler | Skills may interfere |
| Two stages (sequential) | Each stage focuses on one skill | More training time |

**We chose two-stage for best quality.**

### The Pipeline

```
┌─────────────────┐     Stage 1      ┌─────────────────┐
│  Qwen2.5-3B     │ ──────────────→  │  Chat Model     │
│  (Base)         │   50K chat       │  (can chat,     │
│                 │   examples       │   no tools)     │
└─────────────────┘                  └────────┬────────┘
                                              │
                                              │ Stage 2
                                              │ 60K FC examples
                                              ▼
                                     ┌─────────────────┐
                                     │  Final Model    │
                                     │  (chat + tools) │
                                     └─────────────────┘
```

### Stage 1: Chat SFT

**Goal:** Teach base model to follow instructions

**Input:** Base model (outputs random text)
**Data:** 50K UltraChat examples
**Output:** Model that can chat but can't use tools

### Stage 2: Function Calling SFT

**Goal:** Teach chat model to use tools

**Input:** Stage 1 checkpoint (can chat)
**Data:** 60K function calling examples
**Output:** Final model (chat + tools)

### Training Configuration

```python
# Using Unsloth for 2x faster training
# QLoRA for memory efficiency

LoRA Config:
- Rank (r): 16
- Alpha: 16
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

Training:
- Batch size: 4
- Gradient accumulation: 4 (effective batch = 16)
- Learning rate: 2e-4
- Epochs: 1 per stage
- Precision: BF16
- Optimizer: AdamW 8-bit
```

### Expected Results

| Checkpoint | Chat Quality | Function Calling |
|------------|--------------|------------------|
| Base model | ❌ 0% | ❌ 0% |
| After Stage 1 | ✅ Good | ❌ 0% |
| After Stage 2 | ✅ Good | ✅ 90%+ |

---

## Part 5: Training {#part-5-training}

*Coming soon after training completes...*

Topics to cover:
- Unsloth setup on RunPod
- Training logs and metrics
- Loss curves
- Checkpoint management
- Common issues and solutions

---

## Part 6: Evaluation & Results {#part-6-results}

*Coming soon after training completes...*

Topics to cover:
- Function calling accuracy (before/after)
- Chat quality evaluation
- Latency comparison
- Cost analysis
- Model deployment

---

## Key Learnings So Far

### 1. VRAM Calculation
```
VRAM = Params × Bytes + KV Cache + Activations + Overhead

For inference: ~1.2-1.5x model weight size
For QLoRA training: ~1.5-2x model weight size
```

### 2. vLLM is Essential for Production
- 7.5x throughput with continuous batching
- PagedAttention prevents memory fragmentation
- OpenAI-compatible API for easy integration

### 3. Data Format Matters
- Different datasets have different formats
- Preprocessing is 50% of the work
- Always verify format before training

### 4. Base vs Instruct Models
- Instruct models already capable → fine-tuning shows incremental improvement
- Base models are blank slate → fine-tuning shows dramatic improvement
- Choose based on your learning/demo goals

---

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [UltraChat Dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [Salesforce xlam Dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- [Qwen2.5 Models](https://huggingface.co/Qwen)

---

*This blog post is a work in progress. Training results coming soon!*
