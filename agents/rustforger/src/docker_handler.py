import argparse
import sys
import json
import docker
import random
from docker.errors import ImageNotFound, APIError, DockerException
from typing import List, Dict, Optional, Union, Any
import concurrent.futures
import os
from pathlib import Path
from dotenv import load_dotenv

def _find_repo_root() -> Path:
    env_root = os.environ.get("RUSTBENCH_ROOT")
    if env_root:
        return Path(env_root).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() or (parent / "data").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _find_repo_root()
load_dotenv(REPO_ROOT / ".env")


SETUP_CMD = """
echo "Starting setup script..."
mkdir -p /workspace
echo "Created /workspace directory."

cp -r /home/{instance_repo} /workspace/{instance_workspace_name}

# Clean git repo in workspace to avoid future commit leaks
if [ -d /workspace/{instance_workspace_name}/.git ]; then
    rm -rf /workspace/{instance_workspace_name}/.git
fi
git -C /workspace/{instance_workspace_name} init -q
git -C /workspace/{instance_workspace_name} config user.email "rustforger@example.com"
git -C /workspace/{instance_workspace_name} config user.name "RustForger"

# Reuse the task image Cargo registry configuration and lockfile
cargo_config_dir="/workspace/{instance_workspace_name}/.cargo"
mkdir -p "$cargo_config_dir"
if [ -f /testbed/.cargo/config.toml ]; then
    cp /testbed/.cargo/config.toml "$cargo_config_dir/config.toml"
    cp /testbed/.cargo/config.toml /root/.cargo/config.toml
elif [ -f /testbed/.cargo/config ]; then
    cp /testbed/.cargo/config "$cargo_config_dir/config"
    cp /testbed/.cargo/config /root/.cargo/config
fi

if [ -f /testbed/Cargo.lock ]; then
    cp /testbed/Cargo.lock "/workspace/{instance_workspace_name}/Cargo.lock"
fi

# Commit base so git diff works cleanly for patch extraction
git -C /workspace/{instance_workspace_name} add -A
git -C /workspace/{instance_workspace_name} commit -q -m "base"

# Remove git history from /testbed and /home to completely prevent git leakage
rm -rf /testbed/.git /home/{instance_repo}/.git 2>/dev/null || true

# Install prebuilt trace_cli binary directly
mkdir -p /root/.cargo/bin
if [ -f /RTAgent/rustforger-tracer/bin/trace_cli ]; then
    cp /RTAgent/rustforger-tracer/bin/trace_cli /root/.cargo/bin/trace_cli
    chmod +x /root/.cargo/bin/trace_cli
fi

echo "Using host Miniconda rustbench environment..."
export PATH=/home/xupeicai/xpc/envs/rustbench/bin:/root/.cargo/bin:$PATH
export TZ=Asia/Hong_Kong
export RUSTFLAGS="-A warnings"
export USER=root

mkdir -p /home/rust_tracer/ /home/rust_target/
if [ -d "/home/{instance_repo}/target" ] && [ -z "$(ls -A /home/rust_target 2>/dev/null)" ]; then
    cp -rn "/home/{instance_repo}/target/." /home/rust_target/ 2>/dev/null || true
fi
export CARGO_TARGET_DIR=/home/rust_target/
echo "Set CARGO_TARGET_DIR to /home/rust_target/ for shared compilation cache"

cd /workspace

echo "Starting RTA..."
/home/xupeicai/xpc/envs/rustbench/bin/python /RTAgent/src/rustforger_main_tracing.py --instance {instance_id} --max-budget 20.0 --warning-budget 18.0 --max-turns 60
echo "Resolved completed."
"""

ContainerObject = docker.models.containers.Container

def run_command_in_container(
    client: docker.DockerClient,
    image_name: str,
    command: Optional[Union[str, List[str]]] = None,
    container_name: Optional[str] = None,
    privileged: bool = False,
    network_mode: Optional[str] = None,
    volumes: Optional[Dict[str, Dict[str, str]]] = None,
    environment: Optional[Dict[str, str]] = None,
    ports: Optional[Dict[str, Optional[Union[int, str, List[Union[int, str]]]]]] = None,
    stdin_open: bool = False,
    tty: bool = False,
    remove: bool = True,
    detach: bool = False,
    pull_image_if_not_found: bool = True,
    mem_limit: Optional[str] = None
) -> Union[bytes, ContainerObject, None]:
    try:
        if pull_image_if_not_found:
            try:
                client.images.get(image_name)
                print(f"Image '{image_name}' already exists locally.")
            except ImageNotFound:
                print(f"Image '{image_name}' not found locally. Pulling image...")
                client.images.pull(image_name)
                print(f"Image '{image_name}' pulled successfully.")

        run_kwargs: Dict[str, Any] = {
            'image': image_name,
            'command': command,
            'name': container_name,
            'privileged': privileged,
            'network_mode': network_mode,
            'volumes': volumes,
            'environment': environment,
            'ports': ports,
            'stdin_open': stdin_open,
            'tty': tty,
            'remove': remove,
            'detach': detach,
            'mem_limit': mem_limit
        }

        if detach:
            print(f"Starting container from image '{image_name}' in detached mode...")
            container = client.containers.run(**run_kwargs)
            print(f"Container '{container.name if container_name else container.id}' started.")
            return container
        else:
            print(f"Running command in container from image '{image_name}' (foreground)...")
            logs = client.containers.run(**run_kwargs)
            print(f"Command finished in container (from image '{image_name}').")
            if remove:
                print(f"Container was (or will be) removed as per 'remove=True'.")
            return logs

    except ImageNotFound:
        print(f"ERROR: Image '{image_name}' could not be found or pulled.")
        return None
    except APIError as e:
        print(f"Docker API Error: {e}")
        if container_name and "Conflict" in str(e) and f"name \"/{container_name}\" is already in use" in str(e):
            print(f"Hint: A container with name '{container_name}' may already exist.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def _resolve_shared_cargo_cache_dir(repo_key: str) -> str:
    explicit_cache = os.environ.get("RUSTBENCH_CARGO_CACHE_DIR", "").strip()
    if explicit_cache:
        base_dir = explicit_cache
    elif os.environ.get("CARGO_TARGET_DIR", "").strip():
        return os.environ["CARGO_TARGET_DIR"].strip()
    else:
        default_jifang = "/home/xupeicai/xpc/cache/cargo_target"
        if os.path.isdir("/home/xupeicai/xpc"):
            base_dir = default_jifang
        else:
            base_dir = os.path.join(str(Path.home()), ".cache", "rustbench", "cargo_target")
    repo_cache_dir = os.path.join(base_dir, repo_key)
    os.makedirs(repo_cache_dir, exist_ok=True)
    return repo_cache_dir


def _resolve_instance_image_name(instance_info, docker_client=None) -> str:
    """Resolve the canonical evaluation image name for RustForger."""
    # 1. Explicit override from instance record or environment
    if instance_info.get("image_name"):
        return instance_info["image_name"]
    if instance_info.get("task_image"):
        return instance_info["task_image"]
    if os.environ.get("RUSTBENCH_TASK_IMAGE"):
        return os.environ["RUSTBENCH_TASK_IMAGE"].strip()

    org_name = instance_info['repo'].split('/')[0].lower()
    repo_name = instance_info['repo'].split('/')[1].lower()
    pr_number = instance_info.get('pull_number', '')
    short_tag = f"rustbench/{org_name}__{repo_name}:pr-{pr_number}"

    # 2. Dynamic resolution of canonical evaluation image tag
    resolved_tag = None
    try:
        try:
            from swebench.harness.resolve_image import resolve_instance_image_key
        except ImportError:
            candidates = [
                REPO_ROOT / "Rust-bench",
                Path.home() / "xpc" / "Rust-bench",
            ]
            for c in candidates:
                if c.is_dir() and str(c) not in sys.path:
                    sys.path.insert(0, str(c))
            from swebench.harness.resolve_image import resolve_instance_image_key

        config_path = os.environ.get("RUSTBENCH_CONFIG_PATH")
        if not config_path or not os.path.isfile(config_path):
            cfg_candidates = [
                REPO_ROOT / "Rust-bench" / "swebench" / "harness" / "logs" / "config.json",
                Path.home() / "xpc" / "Rust-bench" / "swebench" / "harness" / "logs" / "config.json",
            ]
            for c in cfg_candidates:
                if c.is_file():
                    config_path = str(c)
                    break
        resolved_tag = resolve_instance_image_key(instance_info, config_path=config_path)
    except Exception as e:
        print(f"[docker_handler] Notice: canonical image resolution fallback: {e}")

    # 3. Prefer local image that actually exists: canonical tag > short tag fallback
    if docker_client is not None:
        if resolved_tag:
            try:
                docker_client.images.get(resolved_tag)
                return resolved_tag
            except ImageNotFound:
                pass
        try:
            docker_client.images.get(short_tag)
            return short_tag
        except ImageNotFound:
            pass

    return resolved_tag or short_tag


def process_instance(instance_info):
    """Process a single instance Docker container run"""
    org_name = instance_info['repo'].split('/')[0].lower()
    repo_name = instance_info['repo'].split('/')[1].lower()
    repo_key = f"{org_name}__{repo_name}"
    pr_number = instance_info['pull_number']
    instance_container_name = f"rta-{instance_info['instance_id']}"
    instance_workspace_name = f"{org_name}__{repo_name}__0.1"

    try:
        docker_client = docker.from_env(timeout=18000) 
        docker_client.ping()
        print("Successfully connected to Docker daemon.")
    except DockerException as e:
        print(f"ERROR: Could not connect to Docker daemon. Is Docker running? Details: {e}")
        return

    instance_image_name = _resolve_instance_image_name(instance_info, docker_client=docker_client)
    
    print(instance_workspace_name)
    print(instance_image_name)
    print(instance_container_name)

    is_privileged = True
    network = 'host'

    base_dir = os.path.dirname(os.path.dirname(__file__))
    repo_cache_dir = _resolve_shared_cargo_cache_dir(repo_key)
    data_dir = Path(os.environ.get("RUSTBENCH_DATA_DIR") or (REPO_ROOT / "data"))

    volumes_map = {
        os.path.join(base_dir, 'log'): {'bind': '/RTAgent/log', 'mode': 'rw'},
        os.path.join(base_dir, 'config'): {'bind': '/RTAgent/config', 'mode': 'ro'},
        str(data_dir / 'rb_all_bug_report.json'): {'bind': '/RTAgent/data/rb_all_bug_report.json', 'mode': 'ro'},
        os.path.join(base_dir, 'src'): {'bind': '/RTAgent/src', 'mode': 'ro'},
        os.path.join(base_dir, 'rustforger-tracer'): {'bind': '/RTAgent/rustforger-tracer', 'mode': 'ro'},
        '/home/xupeicai/xpc/envs/rustbench': {'bind': '/home/xupeicai/xpc/envs/rustbench', 'mode': 'ro'},
        '/home/xupeicai/xpc/glibc-2.28-runtime': {'bind': '/home/xupeicai/xpc/glibc-2.28-runtime', 'mode': 'ro'},
        '/home/xupeicai/xpc/gcc-12': {'bind': '/home/xupeicai/xpc/gcc-12', 'mode': 'ro'},
        repo_cache_dir: {'bind': '/home/rust_target', 'mode': 'rw'},
    }

    setup_cmd = SETUP_CMD.format(
        instance_repo=repo_name,
        instance_id=instance_info['instance_id'],
        instance_workspace_name=instance_workspace_name
    )

    container_command = ['/bin/bash', '-c', setup_cmd]

    print(f"\nAttempting to run container '{instance_container_name}' from image '{instance_image_name}'...")
    
    try:
        existing_container = docker_client.containers.get(instance_container_name)
        print(f"Found existing container named '{instance_container_name}'. Attempting to remove it...")
        existing_container.remove(force=True)
        print(f"Successfully removed existing container '{instance_container_name}'.")
    except docker.errors.NotFound:
        print(f"No existing container named '{instance_container_name}' found. Proceeding.")
    except APIError as e:
        print(f"API error while trying to remove existing container '{instance_container_name}': {e}. Proceeding with caution.")

    output_logs = run_command_in_container(
        client=docker_client,
        image_name=instance_image_name,
        command=container_command,
        container_name=instance_container_name,
        privileged=is_privileged,
        network_mode=network,
        volumes=volumes_map,
        environment={
            **{key: os.environ[key] for key in ("BASE_URL", "API_KEY", "MODEL", "GITHUB_TOKEN", "API_LIMIT", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy") if key in os.environ},
            "CARGO_TARGET_DIR": "/home/rust_target",
            "LD_LIBRARY_PATH": "/home/xupeicai/xpc/gcc-12/lib:/home/xupeicai/xpc/gcc-12/x86_64-conda-linux-gnu/lib",
            "LANGCHAIN_OPENAI_TCP_KEEPALIVE": "0",
            "USER": "root",
        },
        stdin_open=True,
        tty=True,
        remove=True,
        detach=False,
        mem_limit='32g'
    )

    if output_logs is not None:
        print(f"\n--- Output from container '{instance_container_name}' ---")
        try:
            print(output_logs.decode('utf-8'))
        except UnicodeDecodeError:
            print("Could not decode output as UTF-8. Raw bytes:")
            print(output_logs)
        print(f"--- End of output from container '{instance_container_name}' ---")
    else:
        print(f"Failed to run container '{instance_container_name}' or it produced no logs.")

    print(f"\nInstance {instance_info['instance_id']} processing finished.")


def load_instances(jsonl_path, filter_id=None, skip_existing_logs=False):
    instances = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if skip_existing_logs:
                log_filename = f"rustforger_log_{data['instance_id']}.json"
                log_path = f"/data/RustAgent/RTAgent/log/{log_filename}"
                if os.path.exists(log_path):
                    continue
            if filter_id:
                if data['instance_id'] in filter_id:
                    instances.append(data)
            else:
                instances.append(data)
    return instances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RustForger Docker Runner")
    parser.add_argument("--api", type=str.lower, choices=("yj", "ollama"),
                        help="API interface from .env to use (default: yj)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the default model for the selected API interface")
    parser.add_argument("--api-limit", type=int, default=None,
                        help="Override API_LIMIT from .env (default: 70)")
    args = parser.parse_args()
    if args.model is not None and not args.model.strip():
        parser.error("--model must not be empty")
    model_override = args.model.strip() if args.model is not None else None
    if args.api_limit is not None:
        if args.api_limit < 1:
            parser.error("--api-limit must be a positive integer")
        os.environ["API_LIMIT"] = str(args.api_limit)
    else:
        limit_value = os.environ.get("API_LIMIT", "70")
        try:
            if int(limit_value) < 1:
                raise ValueError
        except ValueError:
            parser.error("API_LIMIT must be a positive integer")
    if args.api is not None:
        prefix = args.api.upper()
        base_url = os.environ.get(f"{prefix}_BASE_URL")
        api_key = os.environ.get(f"{prefix}_API_KEY")
        model_name = model_override or os.environ.get(f"{prefix}_MODEL")
        if not base_url or not api_key or not model_name:
            parser.error(f"Set {prefix}_BASE_URL, {prefix}_API_KEY, and {prefix}_MODEL (or --model) in .env")
        os.environ["BASE_URL"] = base_url
        os.environ["API_KEY"] = api_key
        os.environ["MODEL"] = model_name
    elif os.environ.get("API_KEY") and (model_override or os.environ.get("MODEL")):
        os.environ["MODEL"] = model_override or os.environ["MODEL"]
    else:
        base_url = os.environ.get("YJ_BASE_URL")
        api_key = os.environ.get("YJ_API_KEY")
        model_name = model_override or os.environ.get("YJ_MODEL")
        if not base_url or not api_key or not model_name:
            parser.error("Set YJ_*/OLLAMA_* (or MODEL and API_KEY) in the repository root .env file")
        os.environ["BASE_URL"] = base_url
        os.environ["API_KEY"] = api_key
        os.environ["MODEL"] = model_name
    worker = 1

    instances = load_instances(str(REPO_ROOT / 'data' / 'rustbench-all.jsonl'), ['tokio-rs__bytes-253'])
    random.shuffle(instances)

    print(f"Processing {len(instances)} instances with {worker} threads")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker) as executor:
        futures = {executor.submit(process_instance, instance): instance['instance_id'] 
                  for instance in instances}
        
        for future in concurrent.futures.as_completed(futures):
            instance_id = futures[future]
            try:
                future.result()
                print(f"Instance {instance_id} completed successfully")
            except Exception as e:
                print(f"Instance {instance_id} generated an exception: {e}")
    
    print("\nAll instances processing completed.")
