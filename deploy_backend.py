"""
Deploy the FastAPI backend to HuggingFace Spaces.
Creates a Docker-based Space and uploads the backend code.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_REPO = "Rakshit1236/trixie-backend"
BACKEND_DIR = Path("hf_backend")

def main():
    api = HfApi()
    user = api.whoami()
    print(f"Logged in as: {user['name']}")

    # Create Space repo
    try:
        api.create_repo(
            repo_id=HF_REPO,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )
        print(f"Space created: https://huggingface.co/spaces/{HF_REPO}")
    except Exception as e:
        print(f"Space creation: {e}")

    # Upload backend files
    files_to_upload = [
        "app.py",
        "requirements.txt",
        "Dockerfile",
    ]

    for fname in files_to_upload:
        fpath = BACKEND_DIR / fname
        if fpath.exists():
            print(f"Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=HF_REPO,
                repo_type="space",
            )
            print(f"  Uploaded!")

    # Upload src/ directory (needed for imports)
    src_dir = Path("src")
    if src_dir.exists():
        for f in src_dir.glob("*.py"):
            print(f"Uploading src/{f.name}...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"src/{f.name}",
                repo_id=HF_REPO,
                repo_type="space",
            )
            print(f"  Uploaded!")

    # Upload config.py
    config_file = Path("config.py")
    if config_file.exists():
        print("Uploading config.py...")
        api.upload_file(
            path_or_fileobj=str(config_file),
            path_in_repo="config.py",
            repo_id=HF_REPO,
            repo_type="space",
        )
        print("  Uploaded!")

    print(f"\nBackend deployed!")
    print(f"URL: https://huggingface.co/spaces/{HF_REPO}")
    print(f"API Docs: https://huggingface.co/spaces/{HF_REPO}/docs")

if __name__ == "__main__":
    main()
