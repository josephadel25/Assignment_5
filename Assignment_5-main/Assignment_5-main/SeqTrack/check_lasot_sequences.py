import os
import csv
import torch
from lib.train.dataset import Lasot

def check_lasot_sequences(target_classes=('book', 'coin')):
    print("🔍 Checking LaSOT dataset for classes:", target_classes)
    dataset = Lasot(split='train')

    # Filter for the target classes again
    filtered_sequences = [seq for seq in dataset.sequence_list if seq.split('-')[0] in target_classes]
    print(f"\n✅ Found {len(filtered_sequences)} sequences for {target_classes}\n")

    results = []
    for seq_name in filtered_sequences:
        seq_id = dataset.sequence_list.index(seq_name)
        seq_path = dataset._get_sequence_path(seq_id)

        try:
            seq_info = dataset.get_sequence_info(seq_id)
            visible = seq_info["visible"]
            total_frames = len(visible)
            visible_count = int(visible.sum().item())

            img_dir = os.path.join(seq_path, "img")
            num_images = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0

            results.append({
                "sequence": seq_name,
                "path": seq_path,
                "total_frames": total_frames,
                "visible_frames": visible_count,
                "images_found": num_images
            })

            print(f"📂 {seq_name} — Frames: {total_frames}, Visible: {visible_count}, Found: {num_images}")

        except Exception as e:
            print(f"❌ Error reading {seq_name}: {e}")

    # Save to CSV
    csv_path = "lasot_sequence_check.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "path", "total_frames", "visible_frames", "images_found"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved to: {os.path.abspath(csv_path)}")

if __name__ == "__main__":
    check_lasot_sequences()
