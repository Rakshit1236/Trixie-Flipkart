"""
Upload pipeline cache and models to HuggingFace dataset repo.
Run this once before deploying the backend.
"""
import os
import pickle
import json
from pathlib import Path
from huggingface_hub import HfApi, login

HF_REPO = "Rakshit1236/trixie-data"
OUTPUT_DIR = Path("output")

def main():
    api = HfApi()
    user = api.whoami()
    print(f"Logged in as: {user['name']}")

    # Create dataset repo if it doesn't exist
    try:
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)
        print(f"Repo ready: {HF_REPO}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload pipeline cache
    cache_path = OUTPUT_DIR / "pipeline_cache.pkl"
    if cache_path.exists():
        print(f"Uploading pipeline_cache.pkl ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(cache_path),
            path_in_repo="pipeline_cache.pkl",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print("  Uploaded!")

    # Upload models
    models_dir = OUTPUT_DIR / "models"
    if models_dir.exists():
        for f in models_dir.glob("*.pkl"):
            print(f"Uploading {f.name} ({f.stat().st_size / 1024:.1f} KB)...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"models/{f.name}",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            print("  Uploaded!")

    print("\nDone! All files uploaded to HuggingFace dataset repo.")
    print(f"Repo URL: https://huggingface.co/datasets/{HF_REPO}")

if __name__ == "__main__":
    main()
