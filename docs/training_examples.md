# Training Examples with Chat Template Applied

Source: `data/stage1_chat/train_qwen_single.jsonl`

Examples: 2

---

## Example 1

**Raw messages:**
```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "content": "Wow, those scenic bus routes all sound amazing! I think I'll definitely have to check out the North Coast 500 and the Isle of Skye.",
    "role": "user"
  },
  {
    "content": "The North Coast 500 and the Isle of Skye are both stunning, and I'm sure you'll have a wonderful time exploring them on the bus. I hope you have a great trip to Scotland!",
    "role": "assistant"
  }
]
```

**After chat template:**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Wow, those scenic bus routes all sound amazing! I think I'll definitely have to check out the North Coast 500 and the Isle of Skye.<|im_end|>
<|im_start|>assistant
The North Coast 500 and the Isle of Skye are both stunning, and I'm sure you'll have a wonderful time exploring them on the bus. I hope you have a great trip to Scotland!<|im_end|>

```

---

## Example 2

**Raw messages:**
```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "content": "I love the imagery in this poem! Could you perhaps add a line or two about the feeling of the sun on your skin when you're alone in nature? It's one of my favorite sensations.",
    "role": "user"
  },
  {
    "content": "Absolutely! Here are a few more lines that incorporate the sensation of the sun on the skin:\n\nThe sun warms my skin with a gentle kiss,\nAs I lose myself in the beauty of this.\nThe world around me is ablaze with light,\nA fiery glow that's a beautiful sight.\n\nThe rays of the sun pierce through the trees,\nEach beam a little moment of reprieve.\nIt's in these moments that I'm truly free,\nWith nothing but nature, the sun, and me.\n\nI bask in the warmth of the morning light,\nFeeling alive, with no worries or fright.\nThe gentle breeze that brushes past my face,\nA sweet caress, a moment of grace.\n\nIn solitude, I am one with nature,\nFeeling every sensation, every texture.\nAnd as the sun sets and the stars come out to play,\nI know that in this world, I'll always find my way.",
    "role": "assistant"
  }
]
```

**After chat template:**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
I love the imagery in this poem! Could you perhaps add a line or two about the feeling of the sun on your skin when you're alone in nature? It's one of my favorite sensations.<|im_end|>
<|im_start|>assistant
Absolutely! Here are a few more lines that incorporate the sensation of the sun on the skin:

The sun warms my skin with a gentle kiss,
As I lose myself in the beauty of this.
The world around me is ablaze with light,
A fiery glow that's a beautiful sight.

The rays of the sun pierce through the trees,
Each beam a little moment of reprieve.
It's in these moments that I'm truly free,
With nothing but nature, the sun, and me.

I bask in the warmth of the morning light,
Feeling alive, with no worries or fright.
The gentle breeze that brushes past my face,
A sweet caress, a moment of grace.

In solitude, I am one with nature,
Feeling every sensation, every texture.
And as the sun sets and the stars come out to play,
I know that in this world, I'll always find my way.<|im_end|>

```

---
# Training Examples with Chat Template Applied

Source: `data/stage2_fc/train_qwen_single.jsonl`

Examples: 3

---

## Example 1

**Raw messages:**
```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant with access to the following tools:\n\n[\n  {\n    \"name\": \"catalog_albums\",\n    \"description\": \"Fetch catalog album data from the Billboard API for a specified week.\",\n    \"parameters\": {\n      \"week\": {\n        \"description\": \"The week for which to fetch the catalog albums chart in the format 'YYYY-MM-DD'. Defaults to '2022-10-08'.\",\n        \"type\": \"str\",\n        \"default\": \"2022-10-08\"\n      }\n    }\n  },\n  {\n    \"name\": \"get_radio_playlist\",\n    \"description\": \"Fetches a radio playlist for a given artist or song using the Spotify API.\",\n    \"parameters\": {\n      \"uri\": {\n        \"description\": \"The URI of the artist or song for which to generate the radio playlist.\",\n        \"type\": \"str\",\n        \"default\": \"spotify:artist:2w9zwq3AktTeYYMuhMjju8\"\n      }\n    }\n  }\n]\n\nWhen you need to use a tool, respond with:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}\n</tool_call>"
  },
  {
    "role": "user",
    "content": "Retrieve the catalog albums from the week of January 15, 2024, and create a radio playlist for the artist Taylor Swift."
  },
  {
    "role": "assistant",
    "content": "<tool_call>\n{\"name\": \"catalog_albums\", \"arguments\": {\"week\": \"2024-01-15\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_radio_playlist\", \"arguments\": {\"uri\": \"spotify:artist:06HL4z0CvFAxyc27GXpf02\"}}\n</tool_call>"
  }
]
```

**After chat template:**
```
<|im_start|>system
You are a helpful assistant with access to the following tools:

[
  {
    "name": "catalog_albums",
    "description": "Fetch catalog album data from the Billboard API for a specified week.",
    "parameters": {
      "week": {
        "description": "The week for which to fetch the catalog albums chart in the format 'YYYY-MM-DD'. Defaults to '2022-10-08'.",
        "type": "str",
        "default": "2022-10-08"
      }
    }
  },
  {
    "name": "get_radio_playlist",
    "description": "Fetches a radio playlist for a given artist or song using the Spotify API.",
    "parameters": {
      "uri": {
        "description": "The URI of the artist or song for which to generate the radio playlist.",
        "type": "str",
        "default": "spotify:artist:2w9zwq3AktTeYYMuhMjju8"
      }
    }
  }
]

When you need to use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call><|im_end|>
<|im_start|>user
Retrieve the catalog albums from the week of January 15, 2024, and create a radio playlist for the artist Taylor Swift.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "catalog_albums", "arguments": {"week": "2024-01-15"}}
</tool_call>
<tool_call>
{"name": "get_radio_playlist", "arguments": {"uri": "spotify:artist:06HL4z0CvFAxyc27GXpf02"}}
</tool_call><|im_end|>

```

---

## Example 2

**Raw messages:**
```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant with access to the following tools:\n\n[\n  {\n    \"name\": \"get_ip_zipcode\",\n    \"description\": \"Retrieves the ZIP code of a given IP address using the ip-api.com API.\",\n    \"parameters\": {\n      \"ip\": {\n        \"description\": \"The IP address to locate.\",\n        \"type\": \"str\"\n      }\n    }\n  },\n  {\n    \"name\": \"calculate_order_total\",\n    \"description\": \"Calculates the total cost of an order based on the items, quantities, and prices.\",\n    \"parameters\": {\n      \"items\": {\n        \"description\": \"A list of item names.\",\n        \"type\": \"List[str]\"\n      },\n      \"quantities\": {\n        \"description\": \"A list of corresponding quantities for each item.\",\n        \"type\": \"List[int]\"\n      },\n      \"prices\": {\n        \"description\": \"A list of corresponding prices for each item.\",\n        \"type\": \"List[float]\"\n      }\n    }\n  },\n  {\n    \"name\": \"place_safeway_order\",\n    \"description\": \"Order specified items from a Safeway location.\",\n    \"parameters\": {\n      \"location\": {\n        \"description\": \"The location of the Safeway store, e.g., 'Palo Alto, CA'.\",\n        \"type\": \"str\"\n      },\n      \"items\": {\n        \"description\": \"List of items to order.\",\n        \"type\": \"list\"\n      },\n      \"quantity\": {\n        \"description\": \"Quantity of each item in the order list.\",\n        \"type\": \"list\"\n      }\n    }\n  },\n  {\n    \"name\": \"get_ip_location\",\n    \"description\": \"Retrieves the latitude and longitude coordinates of a given IP address using the ip-api.com API.\",\n    \"parameters\": {\n      \"ip\": {\n        \"description\": \"The IP address to locate.\",\n        \"type\": \"str\"\n      }\n    }\n  }\n]\n\nWhen you need to use a tool, respond with:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}\n</tool_call>"
  },
  {
    "role": "user",
    "content": "Retrieve the ZIP code for IP 13.13.13.13. Additionally, calculate the total for an order with 2 watches at $100 each and 1 necklace at $50."
  },
  {
    "role": "assistant",
    "content": "<tool_call>\n{\"name\": \"get_ip_zipcode\", \"arguments\": {\"ip\": \"13.13.13.13\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"calculate_order_total\", \"arguments\": {\"items\": [\"watches\", \"necklace\"], \"quantities\": [2, 1], \"prices\": [100, 50]}}\n</tool_call>"
  }
]
```

**After chat template:**
```
<|im_start|>system
You are a helpful assistant with access to the following tools:

[
  {
    "name": "get_ip_zipcode",
    "description": "Retrieves the ZIP code of a given IP address using the ip-api.com API.",
    "parameters": {
      "ip": {
        "description": "The IP address to locate.",
        "type": "str"
      }
    }
  },
  {
    "name": "calculate_order_total",
    "description": "Calculates the total cost of an order based on the items, quantities, and prices.",
    "parameters": {
      "items": {
        "description": "A list of item names.",
        "type": "List[str]"
      },
      "quantities": {
        "description": "A list of corresponding quantities for each item.",
        "type": "List[int]"
      },
      "prices": {
        "description": "A list of corresponding prices for each item.",
        "type": "List[float]"
      }
    }
  },
  {
    "name": "place_safeway_order",
    "description": "Order specified items from a Safeway location.",
    "parameters": {
      "location": {
        "description": "The location of the Safeway store, e.g., 'Palo Alto, CA'.",
        "type": "str"
      },
      "items": {
        "description": "List of items to order.",
        "type": "list"
      },
      "quantity": {
        "description": "Quantity of each item in the order list.",
        "type": "list"
      }
    }
  },
  {
    "name": "get_ip_location",
    "description": "Retrieves the latitude and longitude coordinates of a given IP address using the ip-api.com API.",
    "parameters": {
      "ip": {
        "description": "The IP address to locate.",
        "type": "str"
      }
    }
  }
]

When you need to use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call><|im_end|>
<|im_start|>user
Retrieve the ZIP code for IP 13.13.13.13. Additionally, calculate the total for an order with 2 watches at $100 each and 1 necklace at $50.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_ip_zipcode", "arguments": {"ip": "13.13.13.13"}}
</tool_call>
<tool_call>
{"name": "calculate_order_total", "arguments": {"items": ["watches", "necklace"], "quantities": [2, 1], "prices": [100, 50]}}
</tool_call><|im_end|>

```

---

## Example 3

**Raw messages:**
```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant with access to the following tools:\n\n[\n  {\n    \"name\": \"retreive_collections\",\n    \"description\": \"Retrieves information about collections, including real-time statistics such as floor price.\",\n    \"parameters\": {\n      \"filters\": {\n        \"description\": \"A JSON string representing the filters to apply when retrieving collections. Default is '{\\\"sort\\\":\\\"VOLUME_ONE_DAY\\\",\\\"order\\\":\\\"DESC\\\"}'.\",\n        \"type\": \"str\",\n        \"default\": \"{\\\"sort\\\":\\\"VOLUME_ONE_DAY\\\",\\\"order\\\":\\\"DESC\\\"}\"\n      }\n    }\n  },\n  {\n    \"name\": \"check_similarity\",\n    \"description\": \"Checks if two domains are similar using the Whois Lookup API.\",\n    \"parameters\": {\n      \"domain1\": {\n        \"description\": \"The first domain to compare.\",\n        \"type\": \"str\",\n        \"default\": \"google.com\"\n      },\n      \"domain2\": {\n        \"description\": \"The second domain to compare.\",\n        \"type\": \"str\",\n        \"default\": \"shopify.com\"\n      }\n    }\n  },\n  {\n    \"name\": \"paragraph\",\n    \"description\": \"Generates a specified amount of Lorem Ipsum paragraphs using the RapidAPI Lorem Ipsum API.\",\n    \"parameters\": {\n      \"amount\": {\n        \"description\": \"The number of Lorem Ipsum paragraphs to generate. Defaults to 2.\",\n        \"type\": \"int, optional\",\n        \"default\": \"2\"\n      }\n    }\n  },\n  {\n    \"name\": \"search\",\n    \"description\": \"Searches for entries with specified filters.\",\n    \"parameters\": {\n      \"text\": {\n        \"description\": \"The headword text to search for.\",\n        \"type\": \"str, optional\",\n        \"default\": \"\"\n      },\n      \"page\": {\n        \"description\": \"Page number of results to return (1-indexed). Defaults to 1.\",\n        \"type\": \"int, optional\",\n        \"default\": \"\"\n      },\n      \"number\": {\n        \"description\": \"The grammatical number to search for.\",\n        \"type\": \"str, optional\",\n        \"default\": \"\"\n      },\n      \"monosemous\": {\n        \"description\": \"Whether to only return monosemous entries.\",\n        \"type\": \"bool, optional\",\n        \"default\": \"\"\n      },\n      \"language\": {\n        \"description\": \"The language code of the entry\\u2019s language.\",\n        \"type\": \"str, optional\",\n        \"default\": \"\"\n      },\n      \"analyzed\": {\n        \"description\": \"Whether to search using the language analyzer or to get exact matches only. Defaults to False.\",\n        \"type\": \"bool, optional\",\n        \"default\": \"\"\n      },\n      \"sample\": {\n        \"description\": \"Number of randomly-sampled results to return.\",\n        \"type\": \"int, optional\",\n        \"default\": \"\"\n      },\n      \"pos\": {\n        \"description\": \"The part of speech (POS) to search for.\",\n        \"type\": \"str, optional\",\n        \"default\": \"\"\n      },\n      \"subcategorization\": {\n        \"description\": \"The subcategorization (e.g. countable, transitive) to search for.\",\n        \"type\": \"str, optional\",\n        \"default\": \"\"\n      },\n      \"morph\": {\n        \"description\": \"Whether to search in headword inflections if they exist. Defaults to False.\",\n        \"type\": \"bool, optional\",\n        \"default\": \"\"\n      },\n      \"source\": {\n        \"description\": \"The resource to search within. Defaults to 'global'.\",\n        \"type\": \"str, optional\",\n        \"default\": \"\"\n      },\n      \"gender\": {\n        \"description\": \"The grammatical gender to search for.\",\n        \"type\": \"str, optional\",\n        \"default\": \"\"\n      },\n      \"polysemous\": {\n        \"description\": \"Whether to only return polysemous entries.\",\n        \"type\": \"bool, optional\",\n        \"default\": \"\"\n      }\n    }\n  }\n]\n\nWhen you need to use a tool, respond with:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}\n</tool_call>"
  },
  {
    "role": "user",
    "content": "Retrieve the top NFT collections by volume and check if 'website.com' and 'webdomain.net' are similar."
  },
  {
    "role": "assistant",
    "content": "<tool_call>\n{\"name\": \"retreive_collections\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"check_similarity\", \"arguments\": {\"domain1\": \"website.com\", \"domain2\": \"webdomain.net\"}}\n</tool_call>"
  }
]
```

**After chat template:**
```
<|im_start|>system
You are a helpful assistant with access to the following tools:

[
  {
    "name": "retreive_collections",
    "description": "Retrieves information about collections, including real-time statistics such as floor price.",
    "parameters": {
      "filters": {
        "description": "A JSON string representing the filters to apply when retrieving collections. Default is '{\"sort\":\"VOLUME_ONE_DAY\",\"order\":\"DESC\"}'.",
        "type": "str",
        "default": "{\"sort\":\"VOLUME_ONE_DAY\",\"order\":\"DESC\"}"
      }
    }
  },
  {
    "name": "check_similarity",
    "description": "Checks if two domains are similar using the Whois Lookup API.",
    "parameters": {
      "domain1": {
        "description": "The first domain to compare.",
        "type": "str",
        "default": "google.com"
      },
      "domain2": {
        "description": "The second domain to compare.",
        "type": "str",
        "default": "shopify.com"
      }
    }
  },
  {
    "name": "paragraph",
    "description": "Generates a specified amount of Lorem Ipsum paragraphs using the RapidAPI Lorem Ipsum API.",
    "parameters": {
      "amount": {
        "description": "The number of Lorem Ipsum paragraphs to generate. Defaults to 2.",
        "type": "int, optional",
        "default": "2"
      }
    }
  },
  {
    "name": "search",
    "description": "Searches for entries with specified filters.",
    "parameters": {
      "text": {
        "description": "The headword text to search for.",
        "type": "str, optional",
        "default": ""
      },
      "page": {
        "description": "Page number of results to return (1-indexed). Defaults to 1.",
        "type": "int, optional",
        "default": ""
      },
      "number": {
        "description": "The grammatical number to search for.",
        "type": "str, optional",
        "default": ""
      },
      "monosemous": {
        "description": "Whether to only return monosemous entries.",
        "type": "bool, optional",
        "default": ""
      },
      "language": {
        "description": "The language code of the entry\u2019s language.",
        "type": "str, optional",
        "default": ""
      },
      "analyzed": {
        "description": "Whether to search using the language analyzer or to get exact matches only. Defaults to False.",
        "type": "bool, optional",
        "default": ""
      },
      "sample": {
        "description": "Number of randomly-sampled results to return.",
        "type": "int, optional",
        "default": ""
      },
      "pos": {
        "description": "The part of speech (POS) to search for.",
        "type": "str, optional",
        "default": ""
      },
      "subcategorization": {
        "description": "The subcategorization (e.g. countable, transitive) to search for.",
        "type": "str, optional",
        "default": ""
      },
      "morph": {
        "description": "Whether to search in headword inflections if they exist. Defaults to False.",
        "type": "bool, optional",
        "default": ""
      },
      "source": {
        "description": "The resource to search within. Defaults to 'global'.",
        "type": "str, optional",
        "default": ""
      },
      "gender": {
        "description": "The grammatical gender to search for.",
        "type": "str, optional",
        "default": ""
      },
      "polysemous": {
        "description": "Whether to only return polysemous entries.",
        "type": "bool, optional",
        "default": ""
      }
    }
  }
]

When you need to use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call><|im_end|>
<|im_start|>user
Retrieve the top NFT collections by volume and check if 'website.com' and 'webdomain.net' are similar.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "retreive_collections", "arguments": {}}
</tool_call>
<tool_call>
{"name": "check_similarity", "arguments": {"domain1": "website.com", "domain2": "webdomain.net"}}
</tool_call><|im_end|>

```

---
