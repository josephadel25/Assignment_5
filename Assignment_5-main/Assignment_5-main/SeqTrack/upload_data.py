from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    folder_path=r"D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\Dataset Classes",
    repo_id="ayamohamed2500/seqtrack-checkpoints",   # your repo name
    repo_type="model",                        # or "model"
    allow_patterns="*",
)
