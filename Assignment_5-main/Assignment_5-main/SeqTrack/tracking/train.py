import os
import argparse


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="single",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--seed', type=int, default=42, help='seed for random numbers')

    # ---- START: ADDED FOR AUTOMATION ----
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training from')
    parser.add_argument('--phase', type=str, default="phase_1", help='phase name for Hugging Face upload')
    parser.add_argument('--repo_id', type=str, default=None, help='Hugging Face repo ID for uploading checkpoints')
    parser.add_argument('--hf_train_prefix', type=str, default="member_10_abdelrahman_ahmed/training",
                        help='Subfolder path inside repo to store training artifacts (checkpoints, plots)')
    # ---- END: ADDED FOR AUTOMATION ----

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.mode == "single":
        # ---- START: UPDATED COMMAND ----
        train_cmd = (
            f"python lib/train/run_training.py "
            f"--script {args.script} "
            f"--config {args.config} "
            f"--save_dir {args.save_dir} "
            f"--use_lmdb {args.use_lmdb} "
            f"--seed {args.seed} "
            f"--phase {args.phase} "
            f"--hf_train_prefix \"{args.hf_train_prefix}\" "
            f"{'--repo_id ' + args.repo_id if args.repo_id else ''} "
            f"{'--resume ' + args.resume if args.resume else ''}"
        )
        # ---- END: UPDATED COMMAND ----
    elif args.mode == "multiple":
        if not args.nproc_per_node or str(args.nproc_per_node).lower() == "none":
            # Fallback to single process if nproc_per_node is not provided
            print("nproc_per_node not provided for multiple mode. Falling back to single process.")
            train_cmd = (
                f"python lib/train/run_training.py "
                f"--script {args.script} "
                f"--config {args.config} "
                f"--save_dir {args.save_dir} "
                f"--use_lmdb {args.use_lmdb} "
                f"--seed {args.seed} "
                f"--phase {args.phase} "
                f"--hf_train_prefix \"{args.hf_train_prefix}\" "
                f"{'--repo_id ' + args.repo_id if args.repo_id else ''} "
                f"{'--resume ' + args.resume if args.resume else ''}"
            )
        else:
            train_cmd = (
                f"python -m torch.distributed.launch --nproc_per_node {args.nproc_per_node} lib/train/run_training.py "
                f"--script {args.script} "
                f"--config {args.config} "
                f"--save_dir {args.save_dir} "
                f"--use_lmdb {args.use_lmdb} "
                f"--phase {args.phase} "
                f"--hf_train_prefix \"{args.hf_train_prefix}\" "
                f"{'--repo_id ' + args.repo_id if args.repo_id else ''} "
                f"{'--resume ' + args.resume if args.resume else ''}"
            )
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")

    print(train_cmd)
    os.system(train_cmd)


if __name__ == "__main__":
    main()
