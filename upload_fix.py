"""Re-upload fixed pipeline cache to HuggingFace dataset."""
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
HF_REPO = "Rakshit1236/trixie-data"

cache_path = Path("output/pipeline_cache.json")
print(f"Uploading fixed cache ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)...")
api.upload_file(
    path_or_fileobj=str(cache_path),
    path_in_repo="pipeline_cache.json",
    repo_id=HF_REPO,
    repo_type="dataset",
    commit_message="Fix severity distribution: percentile-based CRITICAL/HIGH/MEDIUM/LOW"
)
print("Done!")
