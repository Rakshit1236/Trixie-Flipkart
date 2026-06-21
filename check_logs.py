from huggingface_hub import HfApi
api = HfApi()

# Get Space info with more details
space = api.space_info("Rakshit1236/trixie-backend")
print(f"Stage: {space.runtime.stage}")
print(f"Image: {getattr(space.runtime, 'image', 'N/A')}")
print(f"Hardware: {getattr(space, 'hardware', 'N/A')}")
print(f"Sdk: {space.sdk}")
print(f"SdkVersion: {getattr(space, 'sdk_version', 'N/A')}")

# Try to get discussion/logs
try:
    discussions = api.get_discussions("Rakshit1236/trixie-backend", repo_type="space")
    for d in list(discussions)[:3]:
        print(f"\nDiscussion: {d.title} - {d.status}")
except Exception as e:
    print(f"Discussions: {e}")
