# Interview Questions: SFT, LoRA, and LLM Training

Based on hands-on experience fine-tuning Qwen2.5-3B for function calling.

---

## Table of Contents
1. [Model Architecture](#model-architecture)
2. [LoRA & QLoRA](#lora--qlora)
3. [Memory & Activations](#memory--activations)
4. [Training Dynamics](#training-dynamics)
5. [Inference Optimizations](#inference-optimizations)
6. [Base Model vs Chat-Tuned Model](#base-model-vs-chat-tuned-model)
7. [Decoding Strategies](#decoding-strategies)
8. [Data & Preprocessing](#data--preprocessing)
9. [SFT Training Loop (Plain PyTorch)](#sft-training-loop-plain-pytorch)
10. [Precision & Numerical Stability](#precision--numerical-stability)
11. [LoRA Configuration Details](#lora-configuration-details)

---

## Model Architecture

### Q: How do you find the architecture details of a model (layers, hidden size, etc.)?

**Answer:**
```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B")
print(config)

# Key attributes:
# config.num_hidden_layers = 36
# config.hidden_size = 2048
# config.num_attention_heads = 16
# config.num_key_value_heads = 2 (GQA)
# config.intermediate_size = 11008
```

---

### Q: What are the 7 main weight matrices in each transformer layer?

**Answer:**

| Matrix | Block | Shape (Qwen 3B) | Purpose |
|--------|-------|-----------------|---------|
| q_proj | Attention | 2048 × 2048 | Projects input → query space |
| k_proj | Attention | 2048 × 256 | Projects input → key space (GQA) |
| v_proj | Attention | 2048 × 256 | Projects input → value space (GQA) |
| o_proj | Attention | 2048 × 2048 | Projects attention output → hidden |
| gate_proj | MLP | 2048 × 11008 | Gating mechanism (SwiGLU) |
| up_proj | MLP | 2048 × 11008 | Expand to intermediate size |
| down_proj | MLP | 11008 × 2048 | Compress back to hidden size |

---

### Q: What is SwiGLU and why is it used?

**Answer:**

SwiGLU is a gated activation function used in modern LLMs (Llama, Qwen).

```python
# Old MLP (GPT-2):
output = down_proj(gelu(up_proj(x)))

# SwiGLU MLP:
gate = silu(gate_proj(x))    # Learns what to let through
up = up_proj(x)               # Actual information
output = down_proj(gate * up) # Gate controls what passes
```

**Why?** The gate learns to suppress irrelevant information and amplify important information, like LSTM gates.

---

### Q: Why are K and V projections smaller in Qwen? (2048 × 256 vs 2048 × 2048)

**Answer:**

Qwen uses **Grouped Query Attention (GQA)**:
- 16 query heads
- Only 2 key/value heads (shared across queries)

```
K_proj output = num_kv_heads × head_dim = 2 × 128 = 256
Q_proj output = num_attention_heads × head_dim = 16 × 128 = 2048
```

**Benefits:**
- Smaller KV cache during inference (8x smaller)
- Faster inference with minimal quality loss

---

### Q: Is hidden_size arbitrary?

**Answer:**

No, there are constraints:

1. **Must be divisible by num_heads**: `head_dim = hidden_size / num_heads` must be integer
2. **Power of 2 preferred**: GPU tensor cores optimized for dims divisible by 8/16
3. **Scaling laws**: Research determines optimal depth vs width ratio for target param count

Common sizes: 512, 768, 1024, 2048, 4096, 8192

---

## LoRA & QLoRA

### Q: Explain LoRA in simple terms.

**Answer:**

Instead of training all weights, LoRA adds small trainable matrices:

```
Original: output = x @ W           (W is frozen)
LoRA:     output = x @ W + x @ (A @ B)  (only A, B trained)

W: (2048, 2048) = 4M params (FROZEN)
A: (2048, 16)   = 32K params (TRAINED)
B: (16, 2048)   = 32K params (TRAINED)

Result: Train 64K params instead of 4M (64x reduction!)
```

---

### Q: What do `r` and `lora_alpha` control?

**Answer:**

```
LoRA scaling factor = alpha / r

r (rank):
- Number of dimensions in adaptation
- Higher r = more capacity = more params
- Typical values: 8, 16, 32, 64

lora_alpha:
- Scaling factor for LoRA updates
- Higher alpha = stronger updates

Example:
r=16, alpha=16  → scaling = 1.0 (balanced)
r=16, alpha=32  → scaling = 2.0 (stronger updates)
r=64, alpha=16  → scaling = 0.25 (weaker updates, more capacity)
```

**Best practice:** When increasing r, increase alpha proportionally to keep ratio stable.

---

### Q: Why is LoRA's B matrix initialized to zeros?

**Answer:**

```python
A = nn.Linear(in_features, r)   # Random init
B = nn.Linear(r, out_features)  # Zero init
```

**Reason:** At initialization, A @ B = 0 (because B is all zeros).

This means:
```
Initial output = x @ W + x @ (A @ B)
               = x @ W + 0
               = x @ W  (unchanged from base model!)
```

**Why this matters:**
- Training starts from the exact pretrained behavior
- No random noise added at start
- Stable training from step 1

**If both were random:**
- Initial output = x @ W + random_noise
- Model starts corrupted
- Training would be unstable

---

### Q: What is QLoRA?

**Answer:**

QLoRA = Quantization + LoRA

```
Base model weights: 4-bit quantized (FROZEN)
LoRA adapters: BF16/FP16 (TRAINED)

Memory comparison for 3B model:
- Full fine-tuning: 3B × 4 bytes = 12 GB (just weights)
- QLoRA: 3B × 0.5 bytes + 50M × 2 bytes = 1.6 GB
```

Enables training 3B models on 8GB GPUs!

---

### Q: Merged vs Separate LoRA - when to use each?

**Answer:**

```
Merged:     W_new = W + A @ B    (compute once, save)
Separate:   output = x @ W + x @ A @ B  (compute each time)
```

| Scenario | Choice | Why |
|----------|--------|-----|
| Production deployment | Merged | Faster inference (1 matmul vs 3) |
| Multiple adapters | Separate | Swap without reloading base model |
| A/B testing | Separate | Quick switching |
| Multi-tenant | Separate | Each customer gets own LoRA |

---

### Q: Loss is stuck at 1.4. Should you increase r, alpha, or both?

**Answer:**

**Both (increase r AND alpha proportionally)**

```
Wrong: r=64, alpha=16
       scaling = 16/64 = 0.25 (weaker updates!)

Right: r=64, alpha=64
       scaling = 64/64 = 1.0 (same strength, more capacity)
```

| Symptom | Fix |
|---------|-----|
| Need more capacity | Increase r AND alpha |
| Learning too slow | Increase alpha (or LR) |
| Unstable/exploding | Decrease alpha (or LR) |
| Overfitting | Decrease r |

---

## Memory & Activations

### Q: What are activations and why do they use so much memory?

**Answer:**

Activations are intermediate outputs saved during forward pass for backward pass.

```
Forward: Input → [Layer 1] → act1 → [Layer 2] → act2 → ... → Output
                              ↓                 ↓
                            SAVE              SAVE
                     (needed for gradients)
```

**Memory calculation (Qwen 3B, batch=4, seq=2048):**

| Component | Size |
|-----------|------|
| Hidden states (36 layers) | ~1.2 GB |
| Attention scores | ~18 GB (without FlashAttention!) |
| MLP intermediates | ~6 GB |

---

### Q: What is gradient checkpointing and what's the tradeoff?

**Answer:**

Instead of storing all activations, store only some ("checkpoints") and recompute the rest during backward pass.

```
Without checkpointing:
- Store: ALL activations
- Memory: HIGH
- Speed: FAST

With checkpointing:
- Store: Every Nth activation
- Recompute: Others during backward
- Memory: LOW (~70% reduction)
- Speed: SLOWER (~30% overhead)
```

**Tradeoff:** Memory vs Compute

```python
# In Unsloth
use_gradient_checkpointing="unsloth"  # Smart checkpointing
```

---

### Q: What is FlashAttention?

**Answer:**

Standard attention creates O(n²) attention matrix:
```
scores = Q @ K^T  → (seq × seq) matrix = 2048 × 2048 = 4M values per head!
```

FlashAttention computes attention in blocks, never materializing full matrix:
```
- Process block by block in fast SRAM
- Discard intermediate results immediately
- O(n) memory instead of O(n²)
```

---

### Q: Memory breakdown for QLoRA training?

**Answer:**

```
| Component              | Memory    |
|------------------------|-----------|
| Model weights (4-bit)  | ~1.5 GB   |
| LoRA adapters (BF16)   | ~100 MB   |
| Optimizer states (8bit)| ~100 MB   |
| Gradients              | ~100 MB   |
| Activations            | ~4-6 GB   |
|------------------------|-----------|
| Total                  | ~6-8 GB   |

Fits on 20GB GPU with room to spare!
```

---

## Training Dynamics

### Q: What does the loss value mean?

**Answer:**

Loss = negative log probability of correct token

```
Loss → Probability of correct token
2.0  → ~13% (e^-2)
1.5  → ~22% (e^-1.5)
1.0  → ~37% (e^-1)
0.5  → ~60% (e^-0.5)
```

**Healthy training signs:**
- Loss decreasing
- Gradient norm stable (0.08-0.15)
- No spikes

**Warning signs:**
- Loss increasing → LR too high
- Loss < 0.3 → overfitting
- Gradient norm exploding → unstable

---

### Q: What's in a typical training log and what to look for?

**Answer:**

```python
{'loss': 1.45, 'grad_norm': 0.12, 'learning_rate': 2e-4, 'epoch': 0.1}
```

| Metric | Healthy | Problem |
|--------|---------|---------|
| loss | Decreasing | Increasing or stuck |
| grad_norm | 0.05-0.2, stable | Spikes or very small |
| learning_rate | Warmup → decay | - |

---

## Inference Optimizations

### Q: What is KV Cache?

**Answer:**

During autoregressive generation, cache K and V from previous tokens:

```
Without cache:
Token 3: Compute K,V for [1,2,3]  ← Recomputed 1,2!

With cache:
Token 3: Compute K,V for [3] only, reuse cached [1,2]

For 100 tokens:
Without: 1+2+3+...+100 = 5,050 computations
With:    100 computations (50x savings!)
```

---

### Q: KV Cache memory calculation?

**Answer:**

```
Size = 2 × num_layers × seq_len × num_kv_heads × head_dim × bytes

Qwen 3B (36 layers, 2 KV heads, 128 head_dim, BF16):
= 2 × 36 × 2048 × 2 × 128 × 2 bytes
= ~75 MB per sequence

10 concurrent users: ~750 MB
```

---

### Q: How to fix "CUDA OOM" during inference generation?

**Answer:**

KV cache is filling up. Solutions:

| Fix | How |
|-----|-----|
| Reduce max_model_len | `LLM(model, max_model_len=2048)` |
| Lower GPU utilization | `gpu_memory_utilization=0.8` |
| Quantize model | AWQ/GPTQ 4-bit |
| Limit concurrent requests | `max_num_seqs=8` |

---

## Base Model vs Chat-Tuned Model

### Q: If we can use sampling parameters (temperature, top_p) to fix repetition, why do we need to fine-tune at all?

**Answer:**

Sampling parameters fix the **style** of generation. Fine-tuning changes the **behavior**.

**Base model behavior (no fine-tuning):**

Base models are trained on next-token prediction on raw text. Given:
```
Input: "What is the capital of France?"
```

A base model might:
- Continue like a document: `"What is the capital of France? This question is often asked in geography..."`
- Generate more questions: `"What is the capital of France? What is the capital of Germany?"`
- Write an essay: `"What is the capital of France? France, officially the French Republic..."`

**It doesn't understand it should ANSWER you.**

**Chat-tuned model behavior:**

After SFT with chat data, the model learns:
1. **Role separation**: System, User, Assistant are distinct roles
2. **Instruction following**: User asks → Assistant answers
3. **When to stop**: Generate EOS token after completing response
4. **Response format**: Direct, helpful, conversational answers

---

### Q: What does sampling vs fine-tuning each fix?

**Answer:**

| Problem | Solution |
|---------|----------|
| Model doesn't answer, just continues text | **Chat SFT** (changes behavior) |
| Model answers but repeats itself | **Sampling parameters** (changes style) |
| Model doesn't stop generating | **EOS token training + stop tokens** |
| Model can't use tools | **Function calling SFT** |
| Model hallucinates facts | **RLHF / DPO** (alignment) |

**Key insight:**
```
Sampling = Controls HOW text is generated (creativity, randomness)
Fine-tuning = Controls WHAT the model learns to do (follow instructions, use tools)
```

---

## Decoding Strategies

### Q: What is greedy decoding and when does it fail?

**Answer:**

**Greedy decoding:** Always pick the highest probability token.

```python
outputs = model.generate(
    input_ids=inputs,
    do_sample=False,  # Greedy!
)
```

**How it works:**
```
Step 1: P(The)=0.3, P(A)=0.2, P(Paris)=0.15 → Pick "The"
Step 2: P(capital)=0.4, P(answer)=0.3 → Pick "capital"
...always pick highest probability
```

**When it fails:**

1. **Repetition loops:**
   ```
   "The answer is 4. The answer is 4. The answer is 4..."
   ```
   Once "The" has high probability, it keeps picking it.

2. **Suboptimal sequences:**
   ```
   Greedy picks: "The" → "capital" → "is" → stuck
   Better path:  "Paris" → "is" → "the" → "capital" (lower start, better end)
   ```

**Use greedy when:** Deterministic output needed (math, code, structured data)

---

### Q: What is temperature and how does it affect generation?

**Answer:**

Temperature scales the logits before softmax:

```python
# Without temperature
probs = softmax(logits)

# With temperature
probs = softmax(logits / temperature)
```

**Effect:**

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.0 | Same as greedy (argmax) | Deterministic |
| 0.3 | Very focused, low variance | Code, factual |
| 0.7 | Balanced creativity | General chat |
| 1.0 | Original distribution | Default |
| 1.5+ | Very random, chaotic | Creative writing |

**Example (logits = [2.0, 1.0, 0.5]):**
```
temp=1.0: probs = [0.59, 0.24, 0.17]  # Moderate spread
temp=0.5: probs = [0.76, 0.17, 0.07]  # More concentrated
temp=2.0: probs = [0.42, 0.33, 0.25]  # More uniform
```

**Low temp:** Model more confident, picks likely tokens
**High temp:** Model more random, explores unlikely tokens

---

### Q: What is nucleus sampling (top-p)?

**Answer:**

**Top-p (nucleus) sampling:** Sample from smallest set of tokens whose cumulative probability ≥ p.

```python
outputs = model.generate(
    do_sample=True,
    top_p=0.9,  # Sample from top 90% probability mass
)
```

**How it works:**

```
Sorted probs: [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
              Token A  B     C     D     E     F     G

top_p=0.9:
Cumulative:  [0.40, 0.65, 0.80, 0.90, ...]
                                   ↑ Stop here!

Sample from: {A, B, C, D} only (covers 90% probability)
Ignore: {E, F, G} (unlikely tokens, might be garbage)
```

**Why top-p > top-k:**

| Method | Problem |
|--------|---------|
| Top-k (always k tokens) | Fixed k doesn't adapt to confidence |
| Top-p (dynamic set) | Adapts: confident = few tokens, uncertain = many |

```
Confident: P = [0.85, 0.10, 0.05]
           top_p=0.9 → only {A} (1 token)
           top_k=3 → {A, B, C} (forces bad options)

Uncertain: P = [0.20, 0.18, 0.17, 0.15, 0.15, 0.15]
           top_p=0.9 → {A, B, C, D, E} (5 tokens, good!)
           top_k=3 → {A, B, C} (misses valid options)
```

---

### Q: What is repetition penalty and how does it work?

**Answer:**

**Repetition penalty:** Divide logits of already-generated tokens by penalty factor.

```python
outputs = model.generate(
    repetition_penalty=1.2,  # Penalize repeated tokens
)
```

**How it works:**

```
Generated so far: ["The", "answer", "is", "4"]

Next token logits (before penalty):
"The": 2.5,  "answer": 2.0,  "is": 1.8,  "Paris": 1.5

After repetition_penalty=1.2:
"The": 2.5/1.2 = 2.08  ← Penalized (already used)
"answer": 2.0/1.2 = 1.67  ← Penalized
"is": 1.8/1.2 = 1.5  ← Penalized
"Paris": 1.5  ← NOT penalized (new token)

Result: "Paris" more likely to be chosen!
```

**Values:**
- 1.0 = No penalty (default)
- 1.1-1.2 = Light penalty (recommended)
- 1.5+ = Strong penalty (might break grammar)

---

### Q: What's the difference between repetition_penalty and frequency/presence penalty?

**Answer:**

| Method | How it works | Used by |
|--------|--------------|---------|
| repetition_penalty | Divides logits by factor | HuggingFace |
| frequency_penalty | Subtracts `count × penalty` | OpenAI |
| presence_penalty | Subtracts flat penalty if token appeared | OpenAI |

**Frequency penalty:**
```
"the" appeared 5 times
frequency_penalty = 0.5
logit_adjustment = -5 × 0.5 = -2.5

More appearances = stronger penalty
```

**Presence penalty:**
```
"the" appeared (any count)
presence_penalty = 0.5
logit_adjustment = -0.5

Binary: appeared or not (count doesn't matter)
```

**Use cases:**
- **frequency_penalty:** Prevent word spam ("very very very very")
- **presence_penalty:** Encourage vocabulary diversity

---

### Q: What are stop tokens and why are they important?

**Answer:**

**Stop tokens:** Tokens that tell generation to stop.

```python
outputs = model.generate(
    eos_token_id=tokenizer.eos_token_id,  # <|im_end|> for Qwen
    # OR
    stop_strings=["<|im_end|>", "\n\nUser:"],
)
```

**Without proper stop tokens:**
```
User: What is 2+2?
Assistant: 4.<|im_end|>
User: What is 3+3?  ← Model generates fake user!
Assistant: 6.
...keeps going forever
```

**With stop tokens:**
```
User: What is 2+2?
Assistant: 4.<|im_end|>  ← STOP!
```

**Why SFT matters:**
- Base model doesn't know when to emit EOS
- Chat SFT teaches: "after answering, generate EOS"
- Model learns from data: every assistant turn ends with EOS

---

### Q: Give me optimal generation settings for different tasks.

**Answer:**

```python
# Factual Q&A (deterministic)
generate(
    do_sample=False,  # Greedy
    max_new_tokens=256,
)

# General chat
generate(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    max_new_tokens=512,
)

# Creative writing
generate(
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    repetition_penalty=1.0,  # Allow some repetition for style
    max_new_tokens=1024,
)

# Code generation
generate(
    do_sample=True,
    temperature=0.2,  # Low randomness
    top_p=0.9,
    max_new_tokens=1024,
)

# Function calling (structured output)
generate(
    do_sample=False,  # Deterministic JSON
    max_new_tokens=256,
)
```

---

### Q: Production ChatGPT repeats itself sometimes. Why?

**Answer:**

Yes! Even production models can repeat because:

1. **Autoregressive nature:** Each token conditions on previous - loops can form

2. **Context window limits:** Long conversations lose early context

3. **Sampling randomness:** Even with penalties, repetition is possible

**How production mitigates:**
- Repetition penalties
- Dynamic temperature adjustment
- Post-processing filters
- Context summarization
- User feedback loops (thumbs down on repetition)

**Key insight:** Repetition is a fundamental LLM challenge, not a training bug. It's reduced, not eliminated.

---

## Data & Preprocessing

### Q: Model outputs `{"name": "func"}` instead of `<tool_call>{"name": "func"}</tool_call>`. What went wrong?

**Answer:**

The `<tool_call>` tags are part of the **training data**, not the tokenizer template.

**Check:** `preprocess_data.py` where assistant responses are formatted:
```python
# Must include the tags!
assistant_content = "<tool_call>\n" + json.dumps(tool_call) + "\n</tool_call>"
```

If tags weren't in training data, model never learned to output them.

---

### Q: Why two-stage training (Chat → Function Calling) instead of mixing?

**Answer:**

**Two-stage advantages:**
1. Each stage focuses on one skill
2. Stage 2 builds on Stage 1's chat ability
3. Lower LR in Stage 2 avoids forgetting chat skills

**Single-stage advantage:**
1. Simpler pipeline

**Best quality:** Two-stage with lower LR for Stage 2.

---

## Quick Reference

### Unsloth Optimizations
- Fused Triton kernels (20-30% faster)
- Optimized RoPE embeddings
- Fused CrossEntropy loss
- Smart gradient checkpointing
- FlashAttention integration

### Training Speed by GPU

| GPU | VRAM | Checkpointing | Speed |
|-----|------|---------------|-------|
| RTX 4000 Ada | 20GB | Required | ~4.5s/step |
| A6000 | 48GB | Optional | ~2.5s/step |
| A100 | 80GB | Not needed | ~1.5s/step |
| H100 | 80GB | Not needed | ~0.8s/step |

### Common LoRA Configs

```python
# Conservative (safe start)
r=16, alpha=16, lr=2e-4

# More capacity
r=32, alpha=32, lr=2e-4

# Aggressive learning
r=16, alpha=32, lr=2e-4
```

---

---

## SFT Training Loop (Plain PyTorch)

### Q: What is the core training loop for LLMs?

**Answer:**

```python
for batch in dataloader:
    # 1. Forward pass
    outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
    loss = outputs.loss

    # 2. Backward pass
    loss.backward()

    # 3. Update weights
    optimizer.step()
    optimizer.zero_grad()
```

That's ALL training is! Everything else (SFTTrainer, LoRA, checkpointing) is optimization around this core.

---

### Q: How does next-token prediction work?

**Answer:**

```
Input:  [The] [cat] [sat] [on] [the]
         ↓     ↓     ↓     ↓     ↓
Model:  pred  pred  pred  pred  pred
         ↓     ↓     ↓     ↓     ↓
Target: [cat] [sat] [on] [the] [mat]

Labels are shifted by 1 position.
Model at position i predicts token at position i+1.
```

```python
# Inside model forward:
shift_logits = logits[..., :-1, :]  # Predictions (all except last)
shift_labels = labels[..., 1:]      # Targets (all except first)
loss = cross_entropy(shift_logits, shift_labels)
```

---

### Q: Why do we only train on assistant responses, not user messages?

**Answer:**

**What we want:**
```
Given: User says "What's the weather?"
Generate: "It's sunny today!"

NOT generate: "What's the weather?" (that's user's job!)
```

**If we trained on user tokens:**
```
Model learns to generate user-like text
At inference: might repeat user patterns instead of responding!
```

**With masking (correct):**
```
input_ids = [USR, Hi, END, ASST, Hello, !, END]
labels    = [-100, -100, -100, -100, Hello, !, END]
              ↑     ↑     ↑     ↑     ↑     ↑   ↑
           IGNORE IGNORE IGNORE IGNORE LEARN LEARN LEARN
```

**Analogy:**
```
User tokens = Question (given, don't memorize)
Assistant tokens = Answer (learn this!)

Like teaching a student: learn answers, not questions.
```

---

### Q: What is -100 in labels?

**Answer:**

```python
# PyTorch CrossEntropyLoss special value
loss = CrossEntropyLoss(ignore_index=-100)

# When label = -100:
# - No loss computed for that position
# - No gradient originates from that position
# - Position is completely ignored for learning
```

---

### Q: Walk through forward and backward pass with label masking.

**Answer:**

**Setup:**
```
input_ids = [USR, Hi, END, ASST, Hello, !, END]
labels    = [-100, -100, -100, -100, Hello, !, END]
```

**Forward Pass:**
```
All positions computed normally through transformer.

Position 0 (USR)   → predicts next token
Position 1 (Hi)    → predicts next token
Position 2 (END)   → predicts next token
Position 3 (ASST)  → predicts next token (should be "Hello")
Position 4 (Hello) → predicts next token (should be "!")
Position 5 (!)     → predicts next token (should be "END")
```

**Loss Computation (shifted):**
```
Position 0 predicts → label[1] = -100  → IGNORED
Position 1 predicts → label[2] = -100  → IGNORED
Position 2 predicts → label[3] = -100  → IGNORED
Position 3 predicts → label[4] = Hello → COMPUTE LOSS ✓
Position 4 predicts → label[5] = !     → COMPUTE LOSS ✓
Position 5 predicts → label[6] = END   → COMPUTE LOSS ✓

Only 3 positions contribute to loss!
```

**Backward Pass:**
```
Loss from positions 3,4,5
        ↓
Gradients flow BACK through all layers
        ↓
All weights updated (because forward used all weights)

Key: Gradients ORIGINATE from non-masked positions only,
     but FLOW THROUGH all positions during backprop.
```

---

### Q: Implement SFT training loop in plain PyTorch.

**Answer:**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def train_sft():
    # Setup
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    model.train()
    model.cuda()

    # Training data
    texts = [
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>",
    ]

    for epoch in range(num_epochs):
        for text in texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"].cuda()

            # Create labels with masking
            labels = input_ids.clone()
            # Find assistant start position and mask everything before
            assistant_start = find_assistant_start(input_ids, tokenizer)
            labels[0, :assistant_start] = -100

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update
            optimizer.step()
            optimizer.zero_grad()
```

---

### Q: What does SFTTrainer add on top of plain PyTorch?

**Answer:**

| Task | Plain PyTorch | SFTTrainer |
|------|---------------|------------|
| Tokenization | Manual | Automatic |
| Label creation | Manual | Automatic |
| Label masking | Manual | Automatic |
| Batching & padding | Manual | Automatic |
| Gradient accumulation | Manual | Automatic |
| Logging | Manual | Built-in |
| Checkpointing | Manual | Built-in |
| Mixed precision | Manual | Built-in |

SFTTrainer is convenience wrapper. Understanding plain PyTorch = understanding what it does.

---

## Precision & Numerical Stability

### Q: Why BF16 instead of FP16 for training?

**Answer:**

```
FP32 (32 bits):  [1 sign] [8 exponent] [23 mantissa]
FP16 (16 bits):  [1 sign] [5 exponent] [10 mantissa]
BF16 (16 bits):  [1 sign] [8 exponent] [7 mantissa]
```

| Format | Range | Precision | Training |
|--------|-------|-----------|----------|
| FP16 | ±65,504 | Better | Can overflow! |
| BF16 | ±3.4×10³⁸ | Worse | Stable |

**Problem with FP16:**
```
Gradient = 80,000 (can happen with bad batch)
FP16 max = 65,504
Result = NaN → training dead!
```

**BF16 handles it:**
```
Gradient = 80,000
BF16 max = 3.4×10³⁸
Result = 80,000 → gradient clipping fixes it
```

**Key insight:** BF16 has same range as FP32, just less precision. For training, range matters more than precision.

---

### Q: What is gradient norm?

**Answer:**

Grad norm = magnitude of all gradients combined (Euclidean length)

```python
grad_norm = √(g₁² + g₂² + g₃² + ... + gₙ²)
```

**What it tells you:**

| grad_norm | Meaning |
|-----------|---------|
| 0.01 - 0.2 | Healthy, stable training |
| < 0.001 | Gradients vanishing (not learning) |
| > 1.0 | Large updates (might be unstable) |
| > 100 | Training probably broken |

**Why monitor:**
```
Step 100: grad_norm = 0.08  ✓
Step 101: grad_norm = 847   ✗ Exploding!
Step 102: loss = NaN        ✗ Dead
```

Grad norm warns you BEFORE loss shows problems.

---

### Q: Why can't we use INT4/INT8 for gradients?

**Answer:**

Gradients need floating point because:

| Requirement | INT | Float |
|-------------|-----|-------|
| Negative values | Limited | ✓ |
| Very small (0.00001) | ✗ Rounds to 0 | ✓ |
| Decimal precision | ✗ | ✓ |

```
Gradient = -0.00003847

INT8: Rounds to 0 → no learning!
BF16: Keeps exact value → learning happens
```

**What uses what:**
- Base weights (stored): INT4 ✓
- Gradients (computed): BF16 required
- Optimizer states: FP32 or INT8

---

### Q: If we clip gradients to max 1.0, why do we still need BF16?

**Answer:**

Overflow happens BEFORE clipping!

```
Step 1: Backprop computes gradient = 80,000
Step 2: FP16 overflows → NaN
Step 3: Clipping sees NaN → can't fix!

vs.

Step 1: Backprop computes gradient = 80,000
Step 2: BF16 stores 80,000 (fine)
Step 3: Clipping reduces to 1.0 → safe!
```

Both BF16 (range) and clipping (safety net) work together.

---

## LoRA Configuration Details

### Q: Why `lora_dropout=0`?

**Answer:**

```python
lora_dropout=0,  # Not 0.1 or 0.05
```

Three reasons:

1. **LoRA is already regularized** - Small rank (r=16) limits capacity

2. **Enables Unsloth speedup** - Dropout breaks fused kernels
   ```
   dropout=0:   Fused operations → fast
   dropout=0.1: Separate operations → slow
   ```

3. **Research shows no benefit** - Dropout doesn't help LoRA empirically

---

### Q: Why `bias="none"`?

**Answer:**

```python
bias="none",  # Not "all" or "lora_only"
```

Options:
- `"none"`: Don't train any biases
- `"all"`: Train all biases
- `"lora_only"`: Only biases in LoRA layers

**Why "none":**

1. **Biases are tiny:**
   ```
   q_proj weight: 2048 × 2048 = 4M params
   q_proj bias:   2048 = 2K params (0.05%!)
   ```

2. **Qwen has no biases anyway** - Modern LLMs often disable them

3. **Simpler** - Fewer things to train

---

### Q: Why lower learning rate for Stage 2?

**Answer:**

```python
# Stage 1
LEARNING_RATE = 2e-4

# Stage 2
LEARNING_RATE = 1e-4  # Half!
```

**Reasons:**

1. **Catastrophic forgetting:**
   ```
   High LR Stage 2: Overwrites chat skills
   Low LR Stage 2:  Preserves chat, adds FC
   ```

2. **Building on good foundation:**
   ```
   Stage 1: Random → Good (needs big updates)
   Stage 2: Good → Better (needs small refinements)
   ```

3. **Stability:**
   - Stage 1 found a good "valley" in loss landscape
   - Low LR explores within the valley
   - High LR might jump out of it

---

### Q: What is gradient accumulation and why use it?

**Answer:**

```python
per_device_train_batch_size=4,
gradient_accumulation_steps=4,
# Effective batch size = 4 × 4 = 16
```

**How it works:**
```
Step 1: Forward batch 1 (4 samples), compute gradients, DON'T update
Step 2: Forward batch 2 (4 samples), accumulate gradients, DON'T update
Step 3: Forward batch 3 (4 samples), accumulate gradients, DON'T update
Step 4: Forward batch 4 (4 samples), accumulate gradients, UPDATE!
```

**Why not just batch_size=16?**

```
batch_size=16:
- Forward 16 samples at once
- Activations for 16 samples in memory
- Might OOM!

batch_size=4, accum=4:
- Forward 4 samples at a time
- Only 4 samples' activations in memory
- Same effective batch, less memory!
```

| Setting | Memory | Effective Batch |
|---------|--------|-----------------|
| batch=16, accum=1 | HIGH | 16 |
| batch=4, accum=4 | LOW | 16 |
| batch=2, accum=8 | LOWER | 16 |

**Tradeoff:** More accumulation = slower (more forward passes), but same result.

---

*Document created during Qwen2.5-3B function calling fine-tuning project.*
