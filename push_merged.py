"""Push already-merged model to HuggingFace"""
from huggingface_hub import HfApi

api = HfApi()
print("Pushing tars-3b-merged to HuggingFace...")
api.upload_folder(
    folder_path="./tars-3b-merged",
    repo_id="shekkari21/tars-3b-merged",
    repo_type="model"
)
print("Done! https://huggingface.co/shekkari21/tars-3b-merged")
