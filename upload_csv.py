"""Upload CSV data to HuggingFace dataset repo."""
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
HF_REPO = "Rakshit1236/trixie-data"

csv_path = Path("data/jan to may police violation_anonymized791b166.csv")
print(f"Uploading CSV ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)...")
api.upload_file(
    path_or_fileobj=str(csv_path),
    path_in_repo="jan to may police violation_anonymized791b166.csv",
    repo_id=HF_REPO,
    repo_type="dataset",
)
print("Done!")
