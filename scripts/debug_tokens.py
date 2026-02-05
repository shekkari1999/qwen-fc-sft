"""
Debug: Check if there's a token ID mismatch between different tokenizer loads
"""
from transformers import AutoTokenizer

def check_tokens():
    print("="*70)
    print("Checking <|im_end|> token ID across different tokenizer loads")
    print("="*70)

    # Method 1: Direct load
    print("\n1. Direct AutoTokenizer load (Qwen/Qwen2.5-3B):")
    tok1 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    id1 = tok1.convert_tokens_to_ids("<|im_end|>")
    print(f"   <|im_end|> = {id1}")
    print(f"   Decode {id1} = {repr(tok1.decode([id1]))}")

    # Method 2: With trust_remote_code
    print("\n2. With trust_remote_code=True:")
    tok2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    id2 = tok2.convert_tokens_to_ids("<|im_end|>")
    print(f"   <|im_end|> = {id2}")
    print(f"   Decode {id2} = {repr(tok2.decode([id2]))}")

    # Method 3: Check eos_token
    print("\n3. Tokenizer special tokens:")
    print(f"   eos_token: {repr(tok1.eos_token)} (id={tok1.eos_token_id})")
    print(f"   pad_token: {repr(tok1.pad_token)} (id={tok1.pad_token_id})")

    # Method 4: Check if <|im_end|> is in vocab directly
    print("\n4. Checking vocabulary:")
    if "<|im_end|>" in tok1.get_vocab():
        print(f"   '<|im_end|>' in vocab: YES, id={tok1.get_vocab()['<|im_end|>']}")
    else:
        print("   '<|im_end|>' in vocab: NO")

    # Method 5: Try encoding a string with <|im_end|>
    print("\n5. Encoding test string 'Hello<|im_end|>':")
    test = "Hello<|im_end|>"
    encoded = tok1.encode(test, add_special_tokens=False)
    print(f"   Encoded: {encoded}")
    print(f"   Decoded tokens:")
    for tid in encoded:
        print(f"     {tid}: {repr(tok1.decode([tid]))}")

    # Method 6: Check added_tokens
    print("\n6. Added tokens encoder (special tokens):")
    for tok, tid in list(tok1.added_tokens_encoder.items())[:10]:
        print(f"   {repr(tok)}: {tid}")

    # Check if im_end is there
    if "<|im_end|>" in tok1.added_tokens_encoder:
        print(f"\n   <|im_end|> is in added_tokens: id={tok1.added_tokens_encoder['<|im_end|>']}")

if __name__ == "__main__":
    check_tokens()
