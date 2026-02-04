# Data Format Conversion for Qwen SFT

This document explains the data format conversions needed for fine-tuning Qwen2.5-3B.

## Target Format (Qwen Chat Format)

All training data must be converted to this format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "User message here"},
    {"role": "assistant", "content": "Assistant response here"}
  ]
}
```

---

## Stage 1: Chat Data (UltraChat)

### Source Format
```json
{
  "prompt": "Create a thank you card...",
  "prompt_id": "4b5aa9c5...",
  "messages": [
    {"role": "user", "content": "Create a thank you card..."},
    {"role": "assistant", "content": "[Your Name]..."},
    {"role": "user", "content": "Can you add more details..."},
    {"role": "assistant", "content": "Sure, I'd be happy..."}
  ]
}
```

### Conversion Needed
✅ **Minimal** - Already has `messages` array with correct roles

**Changes:**
1. Add system prompt if not present
2. Keep only `messages` field

### Output Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Create a thank you card..."},
    {"role": "assistant", "content": "[Your Name]..."},
    {"role": "user", "content": "Can you add more details..."},
    {"role": "assistant", "content": "Sure, I'd be happy..."}
  ]
}
```

---

## Stage 2: Function Calling Data (Salesforce xlam)

### Source Format
```json
{
  "id": 6,
  "query": "Identify the ID of 'Pulp Fiction' and search for torrents related to it.",
  "answers": "[{\"name\": \"get_id\", \"arguments\": {\"q\": \"Pulp Fiction\"}}, {\"name\": \"search_torrents\", \"arguments\": {\"keywords\": \"Pulp Fiction\", \"quantity\": 10}}]",
  "tools": "[{\"name\": \"get_id\", \"description\": \"Fetches the ID of a movie...\", \"parameters\": {...}}, ...]"
}
```

**Note:** `answers` and `tools` are JSON strings, not objects!

### Conversion Needed
🔄 **Significant** - Need to restructure entirely

**Changes:**
1. Parse `tools` JSON string → build system prompt with tool definitions
2. Parse `answers` JSON string → format as `<tool_call>` blocks
3. Create `messages` array with system, user, assistant roles

### Output Format
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to the following tools:\n\n[{\"name\": \"get_id\", ...}, ...]\n\nWhen you need to use a tool, respond with:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "Identify the ID of 'Pulp Fiction' and search for torrents related to it."
    },
    {
      "role": "assistant",
      "content": "<tool_call>\n{\"name\": \"get_id\", \"arguments\": {\"q\": \"Pulp Fiction\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"search_torrents\", \"arguments\": {\"keywords\": \"Pulp Fiction\", \"quantity\": 10}}\n</tool_call>"
    }
  ]
}
```

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 1: CHAT DATA                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  UltraChat Format              Qwen Format                      │
│  ─────────────────            ─────────────                     │
│  {                            {                                 │
│    "prompt": "...",    →        "messages": [                   │
│    "messages": [...]            + {system prompt}               │
│  }                                + existing messages           │
│                                 ]                               │
│                               }                                 │
│                                                                 │
│  Conversion: EASY (just add system prompt)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2: FUNCTION CALLING DATA                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Salesforce Format             Qwen Format                      │
│  ─────────────────            ─────────────                     │
│  {                            {                                 │
│    "query": "...",              "messages": [                   │
│    "tools": "[...]",   →          {system: tools + instructions}│
│    "answers": "[...]"             {user: query}                 │
│  }                                {assistant: <tool_call>...}   │
│                                 ]                               │
│                               }                                 │
│                                                                 │
│  Conversion: COMPLEX (restructure + parse JSON strings)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tool Call Format

Qwen uses `<tool_call>` tags for function calling:

### Single Tool Call
```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
```

### Multiple Tool Calls
```
<tool_call>
{"name": "get_id", "arguments": {"q": "Pulp Fiction"}}
</tool_call>
<tool_call>
{"name": "search_torrents", "arguments": {"keywords": "Pulp Fiction", "quantity": 10}}
</tool_call>
```

---

## Dataset Statistics

| Stage | Dataset | Raw Examples | After Processing |
|-------|---------|--------------|------------------|
| 1 | UltraChat | 200K | 50K (subset) |
| 2 | Salesforce xlam | 60K | ~60K |
| **Total** | | | **~110K** |

---

## Files

```
data/
├── stage1_chat/
│   ├── train.jsonl          # Raw UltraChat data
│   └── train_qwen.jsonl     # Converted to Qwen format
├── stage2_fc/
│   ├── train.jsonl          # Raw Salesforce data
│   └── train_qwen.jsonl     # Converted to Qwen format
```
