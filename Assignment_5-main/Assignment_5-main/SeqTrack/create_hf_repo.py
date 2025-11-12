# ---- create_hf_repo.py ----
from huggingface_hub import HfApi
import os

def create_repo_if_not_exists(repo_id: str, private: bool = True):
    """
    Create a Hugging Face repository if it doesn't already exist.

    Args:
        repo_id (str): The full repo name (e.g. "ayamohamed2500/seqtrack-checkpoints")
        private (bool): Whether the repo should be private
    """
    api = HfApi()

    # Try to check if the repo exists
    try:
        api.repo_info(repo_id)
        print(f"✅ Repo '{repo_id}' already exists.")
    except Exception:
        # If not found, create it
        print(f"🚀 Creating new repo '{repo_id}' on Hugging Face...")
        api.create_repo(repo_id=repo_id, private=private)
        print(f"✅ Repo '{repo_id}' created successfully!")

if __name__ == "__main__":
    # 👇 Change this to your HF username and repo name
    repo_id = "ayamohamed2500/seqtrack-checkpoints"

    # Make sure you're logged in first
    # Run: huggingface-cli login
    create_repo_if_not_exists(repo_id)
