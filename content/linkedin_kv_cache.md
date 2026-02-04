# The KV Cache: The Hidden Bottleneck in LLM Inference

Every time an LLM generates a token, it needs to "remember" all previous tokens. This memory is the KV Cache — and it's often the bottleneck, not compute.

## The Problem

```
KV Cache size = 2 × layers × kv_heads × head_dim × seq_len × batch_size × 2 bytes

For Llama-70B at 8K context:
= 2 × 80 × 8 × 128 × 8192 × 1 × 2 = ~2.7 GB per sequence
```

Scale to 100 concurrent users? That's 270GB just for KV cache.

## The Solutions

**1. Architectural Changes**
- **MHA → MQA → GQA**: Reduce KV heads from 32 → 1 → 8
- Llama 2 70B: 8 KV heads (vs 64 query heads) = 8× less KV memory

**2. Attention Patterns**
- **Sliding Window**: Only store last N tokens (Mistral)
- **Sparse Attention**: Skip tokens based on patterns
- **StreamingLLM**: Keep first + recent tokens only

**3. Quantization**
- **KV Cache Quantization**: FP16 → INT8 → INT4
- 50-75% memory reduction with minimal quality loss

**4. Memory Management**
- **PagedAttention (vLLM)**: Virtual memory for KV cache
- No fragmentation, dynamic allocation
- Enables 2-4× more concurrent requests

**5. Faster Access**
- **FlashAttention**: Fused CUDA kernels
- Reduces memory I/O, not storage
- 2-4× faster attention computation

**6. Speculative Decoding**
- Draft model generates N tokens
- Main model verifies in parallel
- Reduces latency, not memory

## Quick Reference

| Technique | Reduces Memory? | Reduces Latency? |
|-----------|-----------------|------------------|
| GQA/MQA | ✅ | ✅ |
| Sliding Window | ✅ | ✅ |
| KV Quantization | ✅ | ⚠️ |
| PagedAttention | ✅ (utilization) | ✅ |
| FlashAttention | ❌ | ✅ |
| Speculative Decoding | ❌ | ✅ |

---

The best systems combine multiple techniques. vLLM uses PagedAttention + continuous batching. Together with GQA models, you get 10× more throughput than naive PyTorch.

#LLM #MLEngineering #Inference #DeepLearning
