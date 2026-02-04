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

### The Setup Journey (Dependency Hell)

Setting up Unsloth on RunPod wasn't straightforward. Here's what we learned:

**Attempt 1: PyTorch 2.4.0 Template**
```
AttributeError: module 'torch._inductor' has no attribute 'config'
```
Unsloth 2026.x requires torch 2.5+ for `torch._inductor.config`.

**Attempt 2: Upgrade to torch 2.5.1**
```
AttributeError: module 'torch' has no attribute 'int1'
```
Now `torchao` (pulled by transformers) needs torch 2.6+.

**Attempt 3: Upgrade to torch 2.6.0**
```
NotImplementedError: Using datasets = 4.5.0 will cause recursion errors.
```
Unsloth needs datasets ≤ 4.3.0.

**Attempt 4: Fix datasets version**
```
ImportError: cannot import name 'device_synchronize' from 'unsloth_zoo'
```
Unsloth and unsloth-zoo packages out of sync.

**The Solution: Official Unsloth Docker Image**

Stop fighting dependencies. Use `unsloth/unsloth:latest` on RunPod:

```
Template: Unsloth Fine-tuning Template
Image: docker.io/unsloth/unsloth
```

Everything works out of the box. Lesson learned: **use official images for complex ML stacks.**

---

### Understanding the Training Code

Let's break down `train_stage1.py`:

#### 1. Model Loading with QLoRA

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,  # QLoRA magic
)
```

**What's happening:**
- Base model weights loaded in 4-bit (75% memory savings)
- Only ~1.5GB for a 3B model instead of 6GB
- Training happens in BF16, but base stays frozen in 4-bit

#### 2. LoRA Adapter Configuration

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # Rank - dimensions for adaptation
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

**LoRA Math:**
- Original weight: W (3072 × 3072 for hidden_size)
- LoRA adds: W + BA where B is (3072 × 16) and A is (16 × 3072)
- Parameters: 3072 × 16 × 2 = 98K per layer vs 9.4M original
- **~100x fewer trainable parameters**

With 7 target modules across 36 layers:
```
Trainable params: ~50M (1.6% of 3B)
Frozen params: ~3B (98.4%)
```

#### 3. Data Formatting

```python
def format_chat(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}
```

This converts:
```json
{"messages": [{"role": "user", "content": "Hi"}]}
```

To Qwen's format:
```
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
```

#### 4. Training Arguments

```python
TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    warmup_steps=100,
    learning_rate=2e-4,
    bf16=True,
    optim="adamw_8bit",  # 8-bit optimizer states
)
```

**Memory breakdown:**
| Component | Memory |
|-----------|--------|
| Model weights (4-bit) | ~1.5 GB |
| LoRA adapters (BF16) | ~100 MB |
| Optimizer states (8-bit) | ~100 MB |
| Gradients | ~100 MB |
| Activations | ~2-4 GB |
| **Total** | **~6-8 GB** |

This is why QLoRA + 8-bit optimizer lets us train 3B models on 20GB GPUs.

---

### Monitoring Training

#### Loss Interpretation

```
{'loss': 1.531, ...}  # Start
{'loss': 1.361, ...}  # After 30 steps
{'loss': 1.287, ...}  # After 100 steps
{'loss': 1.260, ...}  # Stabilizing
```

**What the numbers mean:**

| Loss | Probability of correct token |
|------|------------------------------|
| 2.0 | ~13% |
| 1.5 | ~22% |
| 1.0 | ~37% |
| 0.5 | ~60% |

**Healthy signs:**
- Loss decreasing (model learning)
- Gradient norm stable (0.08-0.15)
- No sudden spikes

**Warning signs:**
- Loss increasing → LR too high
- Loss < 0.3 → possible overfitting
- Gradient norm exploding → training unstable

#### Learning Rate Schedule

We use linear warmup + cosine decay:

```
LR
0.0002 |        ___
       |      /    \
       |     /      \___
       |    /            \___
       |   /                  \___
0      |__/________________________\____
       0   100                    3125  Steps
         warmup     cosine decay
```

Warmup prevents early instability when gradients are noisy.

---

### Training Timeline

**Hardware:** RTX 4000 Ada (20GB VRAM)

| Stage | Examples | Steps | Time |
|-------|----------|-------|------|
| Stage 1 (Chat) | 50,000 | 3,125 | ~4 hours |
| Stage 2 (FC) | 60,000 | 3,750 | ~4.5 hours |
| **Total** | 110,000 | 6,875 | **~8.5 hours** |

For faster training:
- A6000 (48GB): ~2x faster
- A100 (80GB): ~4x faster
- H100 (80GB): ~6x faster

---

### Checkpointing Strategy

```python
# Save LoRA adapters only (~100MB)
model.save_pretrained(f"{OUTPUT_DIR}/final")

# Save merged model (~6GB) - ready for inference
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/merged",
    tokenizer,
    save_method="merged_16bit"
)
```

**Two outputs:**
1. `/final` - Just adapters, needs base model to use
2. `/merged` - Standalone model, ready for vLLM

We use `/merged` for Stage 2 (build on top) and final deployment.

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
