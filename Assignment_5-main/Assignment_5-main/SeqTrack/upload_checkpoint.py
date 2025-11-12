#!/usr/bin/env python3
"""
Upload a specific checkpoint (or any file) to a Hugging Face repo under a phase folder.
Usage
Upload an existing checkpoint into phase_2:
python upload_checkpoint.py --repo_id USER/seqtrack-checkpoints --phase phase_2 --file checkpoints/phase_1/SEQTRACK_ep0003.pth.tar

Choose a different filename in the repo:
python upload_checkpoint.py --repo_id USER/seqtrack-checkpoints --phase phase_2 --file checkpoints/phase_1/SEQTRACK_ep0003.pth.tar --dest_name SEQTRACK_ep0003_from_phase1.pth.tar

Create the repo if it doesn’t exist (private by default; pass --public for public):
python upload_checkpoint.py --repo_id USER/seqtrack-checkpoints --phase phase_1 --file checkpoints/phase_1/SEQTRACK_ep0001.pth.tar --create_repo

Examples:
  python upload_checkpoint.py \
      --repo_id USER/seqtrack-checkpoints \
      --phase phase_2 \
      --file checkpoints/phase_1/SEQTRACK_ep0003.pth.tar

Notes:
  - Requires `huggingface_hub` and an authenticated token (run: huggingface-cli login)
  - If the repo doesn't exist and --create_repo is passed, it will be created (private by default)
"""

import argparse
import os
import sys
from typing import Optional

try:
    from huggingface_hub import upload_file, HfFolder
except Exception as e:  # pragma: no cover
    print("huggingface_hub is required. Install with: pip install huggingface_hub")
    raise


def _ensure_repo(repo_id: str, create_repo: bool, private: bool) -> None:
    if not create_repo:
        return
    try:
        # Local import to avoid hard dependency when not needed
        from create_hf_repo import create_repo_if_not_exists
    except Exception:
        print("Warning: create_hf_repo.py not found. Skipping repo creation step.")
        return
    try:
        create_repo_if_not_exists(repo_id=repo_id, private=private)
    except Exception as e:
        print(f"Warning: Could not ensure repo '{repo_id}': {e}")


def upload_single_file(
    file_path: str,
    repo_id: str,
    phase: str,
    dest_name: Optional[str] = None,
) -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("Hugging Face token not found. Run `huggingface-cli login` first.")

    filename_in_repo = dest_name if dest_name else os.path.basename(file_path)
    path_in_repo = f"{phase}/{filename_in_repo}"

    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    return path_in_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a specific checkpoint (or file) to Hugging Face under a phase folder")
    parser.add_argument("--repo_id", required=True, type=str, help="Hugging Face repo ID, e.g., USER/seqtrack-checkpoints")
    parser.add_argument("--phase", required=True, type=str, help="Phase folder name in the repo, e.g., phase_1 or phase_2")
    parser.add_argument("--file", required=True, type=str, help="Local path to the file to upload")
    parser.add_argument("--dest_name", type=str, default=None, help="Optional different filename in the repo (default: keep original name)")
    parser.add_argument("--create_repo", action="store_true", help="Create the repo if it does not exist")
    parser.add_argument("--public", action="store_true", help="Create the repo as public when --create_repo is used (default: private)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure repo if requested
    _ensure_repo(repo_id=args.repo_id, create_repo=args.create_repo, private=not args.public)

    try:
        path_in_repo = upload_single_file(
            file_path=args.file,
            repo_id=args.repo_id,
            phase=args.phase,
            dest_name=args.dest_name,
        )
        print(f"✅ Uploaded to {args.repo_id}/{path_in_repo}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


