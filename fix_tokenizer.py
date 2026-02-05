"""Fix tokenizer config and re-push"""
from huggingface_hub import HfApi, hf_hub_download
import json

repo_id = "shekkari21/tars-3b-merged"
api = HfApi()

# Download tokenizer_config.json
print("Downloading tokenizer_config.json...")
config_path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json")

# Read and fix
with open(config_path, "r") as f:
    config = json.load(f)

# Fix extra_special_tokens if it's a list
if "extra_special_tokens" in config:
    est = config["extra_special_tokens"]
    if isinstance(est, list):
        print(f"Fixing extra_special_tokens (was list with {len(est)} items)")
        # Convert list to dict or remove it
        config["extra_special_tokens"] = {}
    elif isinstance(est, dict):
        print("extra_special_tokens already a dict, no fix needed")
else:
    print("No extra_special_tokens field found")

# Save fixed config
fixed_path = "/tmp/tokenizer_config.json"
with open(fixed_path, "w") as f:
    json.dump(config, f, indent=2)

# Upload fixed config
print("Uploading fixed tokenizer_config.json...")
api.upload_file(
    path_or_fileobj=fixed_path,
    path_in_repo="tokenizer_config.json",
    repo_id=repo_id,
    repo_type="model"
)

print(f"Done! Tokenizer fixed at https://huggingface.co/{repo_id}")
