from huggingface_hub import HfApi
api = HfApi()
try:
    user = api.whoami()
    print(f"Logged in as: {user['name']}")
except Exception as e:
    print(f"Not logged in: {e}")
