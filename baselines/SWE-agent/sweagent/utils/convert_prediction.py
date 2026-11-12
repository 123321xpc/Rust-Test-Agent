import argparse
import json
from pathlib import Path


def convert_prediction(
    traj_dir: str | Path,
    instance_id: str,
    output_file: str | Path,
    model_name: str = "swe-agent",
) -> Path:
    traj_path = Path(traj_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    patch_text = ""
    patch_file = traj_path / "patches" / f"{instance_id}.patch"
    all_preds_file = traj_path / "all_preds.jsonl"

    if patch_file.exists():
        patch_text = patch_file.read_text(encoding="utf-8")
    elif all_preds_file.exists():
        for line in all_preds_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("instance_id") == instance_id:
                patch_text = item.get("prediction") or ""
                break

    record = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch_text,
    }
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[convert_prediction] Saved prediction for {instance_id} to {output_path} (patch length: {len(patch_text)})")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert SWE-agent trajectory patch to Rust-bench prediction format")
    parser.add_argument("--traj_dir", required=True, help="Trajectory directory of SWE-agent run")
    parser.add_argument("--instance_id", required=True, help="Instance ID")
    parser.add_argument("--output_file", required=True, help="Output prediction jsonl file path")
    parser.add_argument("--model_name", default="swe-agent", help="Model name for prediction record")
    args = parser.parse_args()

    convert_prediction(args.traj_dir, args.instance_id, args.output_file, args.model_name)


if __name__ == "__main__":
    main()
