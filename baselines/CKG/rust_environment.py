import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

def find_repo_crate_names(repo_path: Path) -> list[str]:
    """Find crate names declared in the repository's Cargo.toml manifests."""
    names = []
    repo = Path(repo_path)
    if not repo.is_dir():
        return names
    for root_dir, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in (".git", "target", ".cargo")]
        if "Cargo.toml" in files:
            m_path = Path(root_dir) / "Cargo.toml"
            try:
                content = m_path.read_text(encoding="utf-8", errors="replace")
                m = re.search(r'''\[package\][^\[]*?name\s*=\s*["']([^"']+)["']''', content, re.DOTALL)
                if m:
                    names.append(m.group(1).strip())
            except Exception:
                pass
    return list(dict.fromkeys(names))


def clean_crate_artifacts(target_dir: Path, crate_names: list[str]):
    """Clean compilation artifacts for the current crate(s) to avoid stale cache pollution across runs."""
    target_path = Path(target_dir)
    if not target_path.is_dir():
        return
    prefixes = set()
    for name in crate_names:
        clean = name.strip()
        if not clean:
            continue
        prefixes.add(clean.replace("_", "-"))
        prefixes.add(clean.replace("-", "_"))

    def _rm_path(p: Path):
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists() or p.is_symlink():
                p.unlink()
        except Exception:
            pass

    for profile in ("debug", "release"):
        pdir = target_path / profile
        if not pdir.is_dir():
            continue
        # 1. Clean .fingerprint
        fp_dir = pdir / ".fingerprint"
        if fp_dir.is_dir():
            for item in list(fp_dir.iterdir()):
                for pfx in prefixes:
                    if item.name == pfx or item.name.startswith(f"{pfx}-") or item.name.startswith(f"lib-{pfx}-"):
                        _rm_path(item)
                        break

        # 2. Clean deps
        deps_dir = pdir / "deps"
        if deps_dir.is_dir():
            for item in list(deps_dir.iterdir()):
                for pfx in prefixes:
                    if (item.name.startswith(f"lib{pfx}") or item.name.startswith(f"{pfx}-")
                            or item.name.startswith(f"{pfx}.")):
                        _rm_path(item)
                        break

        # 3. Clean root of profile dir (compiled binaries, d files)
        for item in list(pdir.iterdir()):
            if item.is_file():
                for pfx in prefixes:
                    if item.name.startswith(f"{pfx}-") or item.name.startswith(f"lib{pfx}-"):
                        _rm_path(item)
                        break

        # 4. Clean incremental & build
        for sub in ("incremental", "build"):
            sdir = pdir / sub
            if sdir.is_dir():
                for item in list(sdir.iterdir()):
                    for pfx in prefixes:
                        if item.name == pfx or item.name.startswith(f"{pfx}-"):
                            _rm_path(item)
                            break


"""Select and verify the benchmark Rust toolchain for one repair."""


def benchmark_toolchain(dataset, instance_id, harness_root):
    matches = []
    with Path(dataset).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                instance = json.loads(line)
                if instance.get("instance_id") == instance_id:
                    matches.append(instance)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one dataset record for {instance_id}; found {len(matches)}")

    sys.path.insert(0, str(Path(harness_root).resolve()))
    try:
        module = importlib.import_module("swebench.harness.test_spec")
        spec = module.make_test_spec(matches[0])
    finally:
        sys.path.pop(0)
    if spec is None:
        raise ValueError(f"Evaluation harness cannot build a test spec for {instance_id}")
    version = getattr(spec, "rust_version", None)
    if not version:
        for command in spec.repo_script_list or []:
            match = re.search(r"rustup default\s+([A-Za-z0-9_.-]+)", command)
            if match:
                version = match.group(1)
                break
    if not version:
        raise ValueError("Evaluation harness did not specify a Rust toolchain")
    return version


def _default_cargo_cache_root() -> Path:
    explicit_cache = os.environ.get("RUSTBENCH_CARGO_CACHE_DIR", "").strip()
    if explicit_cache:
        return Path(explicit_cache)
    jifang_xpc = Path("/home/xupeicai/xpc")
    if jifang_xpc.is_dir():
        return jifang_xpc / "cache" / "cargo_target"
    return Path.home() / ".cache" / "rustbench" / "cargo_target"


def resolve_repo_cache_key(repo_path: Path | None = None, instance_id: str | None = None) -> str:
    if instance_id:
        match = re.match(r"^(.+?)-\d+$", instance_id.strip())
        key = match.group(1) if match else instance_id.strip()
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")
        if sanitized:
            return sanitized
    if repo_path is not None:
        repo = Path(repo_path).resolve()
        git_entry = repo / ".git"
        git_config = None
        if git_entry.is_dir():
            git_config = git_entry / "config"
        elif git_entry.is_file():
            try:
                line = git_entry.read_text(encoding="utf-8", errors="replace").strip()
                if line.startswith("gitdir:"):
                    git_dir = Path(line.split(":", 1)[1].strip())
                    if not git_dir.is_absolute():
                        git_dir = (repo / git_dir).resolve()
                    git_config = git_dir / "config"
                    if not git_config.is_file() and git_dir.parent.name == "worktrees":
                        git_config = git_dir.parent.parent / "config"
            except Exception:
                pass
        if git_config is not None and git_config.is_file():
            try:
                content = git_config.read_text(encoding="utf-8", errors="replace")
                m = re.search(
                    r"url\s*=\s*(?:https?://[^/]+/|git@[^:]+:)([^/]+)/([^/\s]+?)(?:\.git)?\s*$",
                    content,
                    re.MULTILINE,
                )
                if m:
                    return f"{m.group(1).lower()}__{m.group(2).lower()}"
            except Exception:
                pass
        name = repo.name
        if name in {"checkout", "testbed"} and repo.parent.name:
            name = repo.parent.name
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
        if sanitized:
            return sanitized
    return "default"


def _seed_target_dir(target_dir: Path, seed_sources: list[Path]) -> bool:
    if any((target_dir / sub).is_dir() for sub in ("debug", "release")):
        return False
    for seed in seed_sources:
        if seed and seed.is_dir() and any((seed / sub).is_dir() for sub in ("debug", "release")):
            try:
                shutil.copytree(seed, target_dir, dirs_exist_ok=True)
                return True
            except Exception:
                continue
    return False


def configure_cargo_cache(
    env: dict[str, str],
    repo_path: Path | None = None,
    instance_id: str | None = None,
    toolchain: str | None = None,
) -> dict[str, str | bool | None]:
    explicit_target = env.get("CARGO_TARGET_DIR", "").strip()
    cache_root = _default_cargo_cache_root()
    if explicit_target:
        target_dir = Path(explicit_target)
    else:
        repo_key = resolve_repo_cache_key(repo_path=repo_path, instance_id=instance_id)
        if toolchain and toolchain.strip():
            tc_sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", toolchain.strip()).strip("_")
            cache_key = f"{repo_key}__{tc_sanitized}" if tc_sanitized else repo_key
        else:
            cache_key = repo_key
        target_dir = cache_root / cache_key

    target_dir.mkdir(parents=True, exist_ok=True)
    seed_sources = []
    explicit_seed = os.environ.get("RUSTBENCH_CARGO_SEED_DIR", "").strip()
    if explicit_seed:
        seed_sources.append(Path(explicit_seed))
    if repo_path is not None:
        seed_sources.append(Path(repo_path) / "target")
    seeded = _seed_target_dir(target_dir, seed_sources)

    env["CARGO_TARGET_DIR"] = str(target_dir)
    env.setdefault("CARGO_INCREMENTAL", "1")

    # Clean stale compilation artifacts of the current crate to prevent cross-run pollution
    crate_names = find_repo_crate_names(repo_path or Path("."))
    if crate_names:
        clean_crate_artifacts(target_dir, crate_names)

    sccache_bin = shutil.which("sccache")
    if not sccache_bin:
        cargo_sccache = Path.home() / ".cargo" / "bin" / ("sccache.exe" if os.name == "nt" else "sccache")
        if cargo_sccache.is_file():
            sccache_bin = str(cargo_sccache)
    if sccache_bin and not env.get("RUSTC_WRAPPER"):
        env["RUSTC_WRAPPER"] = sccache_bin
        sccache_dir = cache_root / "sccache"
        sccache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("SCCACHE_DIR", str(sccache_dir))

    return {
        "cargo_target_dir": str(target_dir),
        "seeded": seeded,
        "rustc_wrapper": env.get("RUSTC_WRAPPER"),
    }


def prepare_rust_environment(root, benchmark_version, repo_path=None, instance_id=None):
    root = Path(root)
    env = os.environ.copy()
    env.pop("RUSTUP_TOOLCHAIN", None)
    pinned = next((name for name in ("rust-toolchain.toml", "rust-toolchain")
                   if (root / name).is_file()), None)
    forced_version = env.pop("RUSTBENCH_EVAL_TOOLCHAIN_OVERRIDE", "").strip()
    selected_version = forced_version or (benchmark_version if not pinned else None)
    if selected_version:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", selected_version):
            raise ValueError("Invalid benchmark Rust toolchain")
        if forced_version:
            env["RUSTUP_TOOLCHAIN"] = selected_version
        else:
            (root.parent / "rust-toolchain").write_text(selected_version + "\n", encoding="utf-8")
    elif not pinned:
        raise ValueError("A benchmark Rust toolchain is required when the repository does not pin one")
    versions = []
    for executable in ("rustc", "cargo"):
        result = None
        for attempt in range(3):
            try:
                result = subprocess.run([executable, "--version"], cwd=root, env=env,
                                        capture_output=True, text=True, encoding="utf-8",
                                        errors="replace", timeout=60, check=True)
                break
            except (OSError, subprocess.SubprocessError) as exc:
                detail = getattr(exc, "stderr", "") or str(exc)
                low_detail = detail.lower()
                is_transient = attempt < 2 and any(
                    tok in low_detail
                    for tok in ("could not download file", "unsuccessful status code", "timed out", "connection")
                )
                if is_transient:
                    time.sleep(2)
                    continue
                raise RuntimeError(f"Cannot verify {executable} for the evaluation toolchain: {detail}") from exc
        versions.append(result.stdout.strip())
    if selected_version and re.fullmatch(r"\d+\.\d+\.\d+", selected_version):
        actual = re.match(r"rustc\s+(\S+)", versions[0])
        if not actual or actual.group(1) != selected_version:
            raise RuntimeError(f"Expected rustc {selected_version}, but the isolated checkout reports {versions[0]}")
    rustc_match = re.match(r"rustc\s+(\S+)", versions[0])
    resolved_tc = rustc_match.group(1) if rustc_match else benchmark_version
    cache_info = configure_cargo_cache(
        env,
        repo_path=repo_path or root,
        instance_id=instance_id,
        toolchain=resolved_tc,
    )
    info = {
        "source": "evaluation_override" if forced_version else (pinned or "benchmark"),
        "selected": selected_version,
        "rustc": versions[0],
        "cargo": versions[1],
        **cache_info,
    }
    return env, info
