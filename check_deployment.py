from huggingface_hub import HfApi
api = HfApi()

# Check backend space status
try:
    space = api.space_info("Rakshit1236/trixie-backend")
    print(f"Backend Space: {space.id}")
    print(f"  Status: {space.runtime.stage}")
    if hasattr(space, 'sdk'):
        print(f"  SDK: {space.sdk}")
    if hasattr(space.runtime, 'discussion_link'):
        print(f"  Discussion: {space.runtime.discussion_link}")
except Exception as e:
    print(f"Backend check error: {e}")

# Also check if the streamlit space exists
try:
    space2 = api.space_info("Rakshit1236/trixie-flipkart")
    print(f"\nStreamlit Space: {space2.id}")
    print(f"  Status: {space2.runtime.stage}")
except Exception as e:
    print(f"\nStreamlit space check: {e}")
