"""Push already-merged model to HuggingFace"""
from huggingface_hub import HfApi

api = HfApi()
repo_id = "shekkari21/tars-3b-merged"

# Create repo if it doesn't exist
print(f"Creating repo {repo_id}...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

print("Pushing tars-3b-merged to HuggingFace...")
api.upload_folder(
    folder_path="./tars-3b-merged",
    repo_id=repo_id,
    repo_type="model"
)
print(f"Done! https://huggingface.co/{repo_id}")
