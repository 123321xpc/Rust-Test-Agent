import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import docker

from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.test_spec import (
    _create_test_spec_base,
    make_nightly_repo_script_list,
    make_nightly_test_spec,
    make_repo_script_list,
    make_test_spec,
    make_test_spec_nightly_wo_feature,
    make_test_spec_wo_features,
)
from swebench.harness.utils import load_swebench_dataset


def _repo_root() -> Path:
    env_root = os.environ.get("RUSTBENCH_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[3]


def default_config_path() -> str:
    """Return default config.json path bundled with the harness."""
    pkg_cfg = Path(__file__).resolve().parent / "logs" / "config.json"
    if pkg_cfg.is_file():
        return str(pkg_cfg)
    jifang_cfg = Path.home() / "xpc" / "Rust-bench" / "swebench" / "harness" / "logs" / "config.json"
    if jifang_cfg.is_file():
        return str(jifang_cfg)
    return str(pkg_cfg)


def default_dataset_path() -> Optional[str]:
    """Return default rustbench-all.jsonl path if present on disk."""
    candidates = [
        os.environ.get("RUSTBENCH_DATASET_PATH"),
        str(_repo_root() / "data" / "rustbench-all.jsonl"),
        str(Path.home() / "xpc" / "data" / "rustbench-all.jsonl"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def resolve_instance_image_key(inst: Dict[str, Any], config_path: Optional[str] = None) -> str:
    """Resolve the canonical evaluation image key for an instance dict without network calls."""
    resolved_config = config_path or os.environ.get("RUSTBENCH_CONFIG_PATH") or default_config_path()
    configs: Dict[str, Any] = {}
    if resolved_config and os.path.isfile(resolved_config):
        with open(resolved_config, "r", encoding="utf-8") as f:
            configs = json.load(f)

    instance_id = inst.get("instance_id") or inst.get(KEY_INSTANCE_ID)
    is_nightly = (
        instance_id in configs.get("nightly", [])
        or instance_id in configs.get("nightly_wo_features", [])
    )
    repo_gen = make_nightly_repo_script_list if is_nightly else make_repo_script_list
    spec = _create_test_spec_base(
        inst,
        repo_script_generator=repo_gen,
        eval_script_generator=None,
        pre_run_script_generator=None,
        is_nightly_setup=is_nightly,
    )
    if spec is None:
        raise ValueError(f"Could not construct test spec for instance {instance_id}")
    return spec.instance_image_key


def resolve_test_spec(dataset_name: str, instance_id: str, config_path: str):
    dataset = load_swebench_dataset(dataset_name, "test", [instance_id])
    matched = [
        d for d in dataset
        if (d.get("instance_id") or d.get(KEY_INSTANCE_ID)) == instance_id
    ]
    if not matched:
        raise ValueError(f"Instance {instance_id} not found in {dataset_name}")
    inst = matched[0]
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if instance_id in configs.get("default", []):
        spec = make_test_spec(inst)
    elif instance_id in configs.get("nightly", []):
        spec = make_nightly_test_spec(inst)
    elif instance_id in configs.get("test", []):
        spec = make_test_spec_wo_features(inst)
    elif instance_id in configs.get("nightly_wo_features", []):
        spec = make_test_spec_nightly_wo_feature(inst)
    else:
        spec = make_test_spec(inst)
    return spec


def resolve_image_name(
    instance_or_id: Union[Dict[str, Any], str],
    dataset_name: Optional[str] = None,
    config_path: Optional[str] = None,
    build_if_missing: bool = False,
    client: Optional[docker.DockerClient] = None,
) -> str:
    """Resolve the canonical Docker image name for an instance dict or instance_id."""
    if isinstance(instance_or_id, dict):
        inst = instance_or_id
        instance_id = inst.get("instance_id") or inst.get(KEY_INSTANCE_ID)
        image_name = resolve_instance_image_key(inst, config_path=config_path)
    else:
        instance_id = str(instance_or_id)
        resolved_dataset = dataset_name or default_dataset_path()
        if not resolved_dataset:
            raise ValueError(f"dataset_name is required when resolving image by instance_id: {instance_id}")
        dataset = load_swebench_dataset(resolved_dataset, "test", [instance_id])
        matched = [
            d for d in dataset
            if (d.get("instance_id") or d.get(KEY_INSTANCE_ID)) == instance_id
        ]
        if not matched:
            raise ValueError(f"Instance {instance_id} not found in {resolved_dataset}")
        image_name = resolve_instance_image_key(matched[0], config_path=config_path)
        dataset_name = resolved_dataset

    if build_if_missing:
        cli = client or docker.from_env()
        existing = cli.images.list(filters={"reference": image_name})
        if not existing:
            if not dataset_name:
                raise ValueError("dataset_name is required when build_if_missing=True")
            cfg = config_path or default_config_path()
            print(f"[resolve_image] Image {image_name} not found, building via run_evaluation...", file=sys.stderr)
            cmd = [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                dataset_name,
                "--predictions_path",
                "gold",
                "--instance_ids",
                instance_id,
                "--config_path",
                cfg,
                "--run_id",
                "build_image_init",
                "--build_image_only",
                "1",
            ]
            subprocess.run(cmd, check=True)

    return image_name


def main():
    parser = argparse.ArgumentParser(description="Resolve task Docker image for Rust-bench instance")
    parser.add_argument("--dataset_name", default=None, help="Path to Rust-bench dataset JSONL (optional if default exists)")
    parser.add_argument("--instance_id", default=None, help="Instance ID")
    parser.add_argument("--instance_stdin", action="store_true", help="Read instance JSON object from stdin")
    parser.add_argument("--config_path", default=None, help="Path to config.json (optional, defaults to bundled config)")
    parser.add_argument("--build_if_missing", action="store_true", help="Build image if not found in local Docker")
    args = parser.parse_args()

    cfg = args.config_path or default_config_path()
    if args.instance_stdin:
        inst = json.load(sys.stdin)
        image_name = resolve_instance_image_key(inst, config_path=cfg)
    else:
        if not args.instance_id:
            raise ValueError("--instance_id or --instance_stdin is required")
        image_name = resolve_image_name(
            args.instance_id,
            dataset_name=args.dataset_name,
            config_path=cfg,
            build_if_missing=args.build_if_missing,
        )
    print(image_name)


if __name__ == "__main__":
    main()
