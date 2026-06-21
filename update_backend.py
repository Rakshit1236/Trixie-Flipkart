"""Upload updated backend + JSON cache to HuggingFace."""
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
HF_REPO = "Rakshit1236/trixie-data"
SPACE_REPO = "Rakshit1236/trixie-backend"

# Upload JSON cache
print("Uploading JSON cache...")
api.upload_file(
    path_or_fileobj=str(Path("output/pipeline_cache.json")),
    path_in_repo="pipeline_cache.json",
    repo_id=HF_REPO,
    repo_type="dataset",
)
print("  Done!")

# Upload updated backend
print("\nUpdating backend...")
for local, remote in [
    ("hf_backend/app.py", "app.py"),
    ("hf_backend/Dockerfile", "Dockerfile"),
    ("hf_backend/requirements.txt", "requirements.txt"),
]:
    if Path(local).exists():
        print(f"  Uploading {remote}...")
        api.upload_file(
            path_or_fileobj=str(Path(local)),
            path_in_repo=remote,
            repo_id=SPACE_REPO,
            repo_type="space",
        )

print("\nDone! Space will rebuild.")
print(f"URL: https://huggingface.co/spaces/{SPACE_REPO}")
