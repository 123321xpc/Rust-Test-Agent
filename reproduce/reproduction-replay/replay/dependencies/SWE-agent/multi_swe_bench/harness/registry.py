import threading
import json
from typing import Optional
from multi_swe_bench.harness.constant import KEY_INSTANCE_ID
import os
from pathlib import Path

data_registry = dict()
lock = threading.Lock()


def merge_benchmark_metadata(pr_data: list[dict], metadata_data: list[dict]) -> int:
    """Fill missing task toolchain metadata from full benchmark records."""
    metadata_by_id = {
        row.get(KEY_INSTANCE_ID): row
        for row in metadata_data
        if row.get(KEY_INSTANCE_ID)
    }
    matched = 0
    metadata_fields = ("created_at", "rust_version", "rustc_version", "rustc")
    for row in pr_data:
        benchmark_row = metadata_by_id.get(row.get(KEY_INSTANCE_ID))
        if benchmark_row is None:
            continue
        matched += 1
        for field_name in metadata_fields:
            if row.get(field_name) in (None, "") and benchmark_row.get(field_name) not in (None, ""):
                row[field_name] = benchmark_row[field_name]

        benchmark_metadata = benchmark_row.get("metadata")
        input_metadata = row.get("metadata")
        if isinstance(benchmark_metadata, dict) or isinstance(input_metadata, dict):
            row["metadata"] = {
                **(benchmark_metadata if isinstance(benchmark_metadata, dict) else {}),
                **(input_metadata if isinstance(input_metadata, dict) else {}),
            }
    return matched


def find_benchmark_metadata_file(registered_data_path: str) -> Optional[Path]:
    configured_path = os.environ.get("RUSTBENCH_METADATA_FILE")
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    candidates.append(Path.home() / "xpc" / "data" / "rustbench-all.jsonl")
    task_path = Path(registered_data_path).resolve()
    if len(task_path.parents) >= 4:
        candidates.append(task_path.parents[3] / "data" / "rustbench-all.jsonl")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None

def register_data(pr_file):
    global data_registry

    
    current_script_path = os.path.abspath(__file__)
    
    current_directory = os.path.dirname(current_script_path)
    
    relative_file_path = os.path.join('../../', pr_file)
    
    registered_data_path = os.path.join(current_directory, relative_file_path)

    if data_registry is not None and len(data_registry.keys()) > 0:
        print(f"data {registered_data_path} is already registered.")
        return

    
    lock.acquire()
    try:
        
        with open(registered_data_path, "r") as f:
            data_list = [json.loads(line) for line in f]

        metadata_path = find_benchmark_metadata_file(registered_data_path)
        if metadata_path is not None:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata_list = [json.loads(line) for line in f]
            matched = merge_benchmark_metadata(data_list, metadata_list)
            print(
                f"Merged benchmark toolchain metadata for {matched}/{len(data_list)} tasks "
                f"from {metadata_path}."
            )
        else:
            print(
                "No Rust-bench metadata file found; set RUSTBENCH_METADATA_FILE "
                "to enable created_at-based toolchain selection."
            )

        data_registry.update({data[KEY_INSTANCE_ID]: data for data in data_list})
    finally:
        
        lock.release()
    print(f"register data {registered_data_path} success.")

def get_data(instance_id: str):
    if data_registry is None or len(data_registry.keys()) == 0:
        register_data()
    data = data_registry.get(instance_id)
    if data is None:
        raise RuntimeError(f"data {instance_id} not found in data registry.")

    return data
