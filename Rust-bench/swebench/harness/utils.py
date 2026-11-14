import json
import os
from pathlib import Path
import re
import requests
import tomlkit
from tomlkit import parse, dumps
from argparse import ArgumentTypeError
from datasets import Dataset, load_dataset
from functools import cache
from typing import cast
import toml
from typing import Dict, List
from swebench.harness.constants import (
    SWEbenchInstance,
    MAP_REPO_TO_ENV_YML_PATHS,
    MAP_REPO_TO_REQS_PATHS,
    NON_TEST_EXTS,
    SWE_BENCH_URL_RAW,
    KEY_INSTANCE_ID,
    NON_OSDK_CRATES,
    OSDK_CRATES
)



def load_swebench_dataset(name="princeton-nlp/SWE-bench", split="test", instance_ids=None) -> list[SWEbenchInstance]:
    """
    Load SWE-bench dataset from Hugging Face Datasets or local .json/.jsonl file
    """
    # check that all instance IDs are in the dataset
    if instance_ids:
        instance_ids = set(instance_ids)
    # Load from local .json/.jsonl file
    if name.endswith(".json") or name.endswith(".jsonl"):
        instances = [
            cast(SWEbenchInstance, json.loads(instance))
            for instance in Path(name).read_text(encoding="utf-8").split("\n")
            if instance
        ]
        if instance_ids:
            instances = [
                i for i in instances
                if (i.get(KEY_INSTANCE_ID) or i.get("instance_id")) in instance_ids
            ]
        return instances

    # Load from Hugging Face Datasets
    if name.lower() in {"swe-bench", "swebench", "swe_bench"}:
        name = "princeton-nlp/SWE-bench"
    elif name.lower() in {"swe-bench-lite", "swebench-lite", "swe_bench_lite", "swe-bench_lite", "lite"}:
        name = "princeton-nlp/SWE-bench_Lite"
    dataset = cast(Dataset, load_dataset(name, split=split))
    # print(f"Loaded {len(dataset)} instances from {name} dataset")
    if instance_ids:
        return [
            cast(SWEbenchInstance, instance)
            for instance in dataset
            if instance[KEY_INSTANCE_ID] in instance_ids
        ]
    return [cast(SWEbenchInstance, instance) for instance in dataset]


### MARK - Patch Correction
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
PATCH_FILE_PATTERN = re.compile(r"\-\-\-\s+a\/(?:.+)\n\+\+\+\s+b\/(?:.+)")
PATCH_HUNK_PATTERN = re.compile(
    r"\@\@\s+\-(\d+),?(\d*)\s+\+(\d+),?(\d*)\s+\@\@(.+?)(?=diff\ |\-\-\-\ a\/|\@\@\ \-|\Z)",
    re.DOTALL,
)


def _repair_single_file_diff(file_diff: str) -> str:
    lines = file_diff.splitlines()
    diff_git_line = None
    old_file_line = None
    new_file_line = None
    meta_lines = []
    hunks = []
    current_hunk = []
    in_hunk = False

    for line in lines:
        if line.startswith("diff --git "):
            diff_git_line = line
        elif line.startswith("--- "):
            old_file_line = line
        elif line.startswith("+++ "):
            new_file_line = line
        elif line.startswith("@@ "):
            in_hunk = True
            if current_hunk:
                hunks.append(current_hunk)
                current_hunk = []
            current_hunk.append(line)
        elif in_hunk:
            current_hunk.append(line)
        else:
            if not in_hunk:
                meta_lines.append(line)

    if current_hunk:
        hunks.append(current_hunk)

    if not hunks:
        return file_diff

    if not diff_git_line and old_file_line and new_file_line:
        old_m = re.search(r"---\s+(?:a/)?(.*)", old_file_line)
        new_m = re.search(r"\+\+\+\s+(?:b/)?(.*)", new_file_line)
        old_p = old_m.group(1).strip() if old_m else ""
        new_p = new_m.group(1).strip() if new_m else ""
        if old_p and new_p:
            src = old_p if old_p == "/dev/null" else f"a/{old_p}"
            dst = new_p if new_p == "/dev/null" else f"b/{new_p}"
            diff_git_line = f"diff --git {src} {dst}"

    repaired_hunks = []
    total_delta = 0
    hunk_header_re = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$")

    for h in hunks:
        h_head = h[0]
        h_body = h[1:]
        m = hunk_header_re.match(h_head)
        if not m:
            repaired_hunks.append("\n".join(h))
            continue

        pre_start = int(m.group(1))
        trailing_suffix = m.group(5) or ""

        first_mod = None
        last_mod = None
        for idx, line in enumerate(h_body):
            if line.startswith("+") or line.startswith("-"):
                if first_mod is None:
                    first_mod = idx
                last_mod = idx

        if first_mod is None:
            continue

        ctx_keep = 3
        start_idx = max(0, first_mod - ctx_keep)
        end_idx = min(len(h_body), last_mod + ctx_keep + 1)
        trimmed_body = h_body[start_idx:end_idx]

        ctx_count = 0
        del_count = 0
        add_count = 0
        clean_lines = []
        for line in trimmed_body:
            if not line:
                norm_line = " "
            elif line[0] in ("+", "-"):
                norm_line = line
            elif line.startswith(" "):
                norm_line = line
            elif line.startswith("\\ No newline at end of file"):
                norm_line = line
            else:
                norm_line = " " + line

            if norm_line.startswith("-"):
                del_count += 1
            elif norm_line.startswith("+"):
                add_count += 1
            elif norm_line.startswith(" "):
                ctx_count += 1
            clean_lines.append(norm_line.rstrip())

        adj_pre_start = pre_start + start_idx
        adj_pre_len = ctx_count + del_count
        adj_post_start = adj_pre_start + total_delta
        adj_post_len = ctx_count + add_count
        total_delta += (adj_post_len - adj_pre_len)

        new_head = f"@@ -{adj_pre_start},{adj_pre_len} +{adj_post_start},{adj_post_len} @@{trailing_suffix}"
        repaired_hunks.append(new_head + "\n" + "\n".join(clean_lines))

    if not repaired_hunks:
        return file_diff

    out = []
    if diff_git_line:
        out.append(diff_git_line)
    for mline in meta_lines:
        if mline.startswith(("old mode", "new mode", "deleted file mode", "new file mode", "index ")):
            out.append(mline)
    if old_file_line:
        out.append(old_file_line)
    if new_file_line:
        out.append(new_file_line)
    out.extend(repaired_hunks)
    return "\n".join(out)


def extract_minimal_patch(model_patch: str) -> str:
    if not model_patch or not model_patch.strip():
        return ""

    text = model_patch.replace("\r\n", "\n")

    # Extract diff within markdown fences if present
    md_matches = re.findall(r"```(?:diff|patch)?\s*\n(.*?)\n```", text, re.DOTALL)
    if md_matches:
        text = "\n".join(md_matches)

    # Locate diff start, dropping conversational prefix
    start_match = re.search(r"(?m)^(?:diff --git|--- (?:a/|/dev/null))", text)
    if start_match:
        text = text[start_match.start():]

    lines = text.splitlines()
    diff_blocks = []
    current_block = []

    for line in lines:
        is_file_start = (
            line.startswith("diff --git ") or
            ((line.startswith("--- a/") or line.startswith("--- /dev/null")) and
             (not current_block or any(l.startswith("+++ b/") or l.startswith("+++ /dev/null") for l in current_block)))
        )
        if is_file_start and current_block:
            diff_blocks.append("\n".join(current_block))
            current_block = []
        current_block.append(line)

    if current_block:
        diff_blocks.append("\n".join(current_block))

    repaired_files = []
    for block in diff_blocks:
        repaired = _repair_single_file_diff(block)
        if repaired:
            repaired_files.append(repaired)

    if not repaired_files:
        return model_patch

    return "\n".join(repaired_files).rstrip() + "\n"


def has_attribute_or_import_error(log_before):
    """
    Check to see if Attribute/Import-prefix is in log text

    Args:
        log_before (str): Validation log text before patch application
    """
    log_before = log_before.lower()

    if any([x in log_before for x in ["attribute", "import"]]):

        def get_lines_with_word(text, target_word):
            # Function to extract line(s) that contains target_word
            text, target_word = text.lower(), target_word.lower()
            lines, hits = text.split("\n")[::-1], []
            for line in lines:
                if target_word in line:
                    hits.append(line)
            return hits

        # Get line with Attribute/Import error
        lines_1 = get_lines_with_word(log_before, "attribute")
        lines_2 = get_lines_with_word(log_before, "import")
        lines_1 = " ".join(lines_1)
        lines_2 = " ".join(lines_2)

        if any([(x in lines_1 or x in lines_2) for x in ["error", "fail"]]):
            return True
    return False


@cache
def get_environment_yml_by_commit(repo: str, commit: str, env_name: str) -> str:
    for req_path in MAP_REPO_TO_ENV_YML_PATHS[repo]:
        reqs_url = os.path.join(SWE_BENCH_URL_RAW, repo, commit, req_path)
        reqs = requests.get(reqs_url)
        if reqs.status_code == 200:
            break
    else:
        raise ValueError(
            f"Could not find environment.yml at paths {MAP_REPO_TO_ENV_YML_PATHS[repo]} for repo {repo} at commit {commit}"
        )

    lines = reqs.text.split("\n")
    cleaned = []
    for line in lines:
        # Rename environment to given name
        if line.startswith("name:"):
            cleaned.append(f"name: {env_name}")
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def get_environment_yml(instance: SWEbenchInstance, env_name: str) -> str:
    """
    Get environment.yml for given task instance

    Args:
        instance (dict): SWE Bench Task instance
        env_name (str): Rename retrieved environment.yml to this name
    Returns:
        environment.yml (str): Returns environment.yml as string
    """
    # Attempt to find environment.yml at each path based on task instance's repo

    commit = (
        instance["environment_setup_commit"]
        if "environment_setup_commit" in instance
        else instance["base_commit"]
    )

    return get_environment_yml_by_commit(instance["repo"], commit, env_name)

def is_workspace_cargo_toml(f):

    try:
        
        cargo_data = toml.load(f)
        
        workspace = cargo_data.get('workspace', {})
        members = workspace.get('members', [])
        
        return bool(workspace) and isinstance(members, list) and len(members) > 0
    except Exception as e:
        print(f"Error reading cargo.toml: {e}")
        return False


@cache
def get_requirements_by_commit(repo: str, commit: str,req_path) -> str:
    reqs_url = os.path.join(SWE_BENCH_URL_RAW, repo, commit, req_path)
    reqs = requests.get(reqs_url)
    original_cargo_toml = reqs.text
    return original_cargo_toml




def clean_cargo_toml(cargo_toml_content):

    try:
        cargo_data = toml.loads(cargo_toml_content)
    except toml.TomlDecodeError as e:
        raise ValueError(f"无法解析 Cargo.toml 内容。错误: {e}")
    
    
    sections_to_keep = ['dependencies', 'dev-dependencies', 'build-dependencies','features']

    
    package_keys_to_keep = ['name', 'version', 'edition']

    cleaned_data = {}

    
    if 'package' in cargo_data:
        package = cargo_data['package']
        cleaned_package = {}
        for key in package_keys_to_keep:
            if key in package:
                cleaned_package[key] = package[key]
        if cleaned_package:
            
            cleaned_package['version']='0.0.0'
            cleaned_data['package'] = dict(sorted(cleaned_package.items()))
    
    
    for section in sections_to_keep:
        if section in cargo_data:
            
            cleaned_section = dict(sorted(cargo_data[section].items()))
            cleaned_data[section] = cleaned_section

    
    cleaned_cargo_toml = toml.dumps(cleaned_data)

    
    cleaned_cargo_toml = '\n'.join(line.rstrip() for line in cleaned_cargo_toml.splitlines())

    return cleaned_cargo_toml

def clean_workspace_cargo_toml(cargo_toml_content: str) -> str:

    try:
        cargo_data = toml.loads(cargo_toml_content)
    except toml.TomlDecodeError as e:
        raise ValueError(f"无法解析 Cargo.toml 内容。错误: {e}")

    cleaned_data = {}

    if 'workspace' in cargo_data:
        workspace = cargo_data['workspace']
        cleaned_workspace = {}

        
        workspace_keys_to_keep = ['members', 'exclude', 'resolver']
        for key in workspace_keys_to_keep:
            if key in workspace:
                cleaned_workspace[key] = workspace[key]

        
        if 'dependencies' in workspace:
            dependencies = workspace['dependencies']
            cleaned_dependencies = {}

            for dep_name, dep_info in dependencies.items():
                
                if isinstance(dep_info, dict):
                    
                    cleaned_dep_info = dep_info.copy()
                    cleaned_dep_info['version'] = "0.0.0"
                    cleaned_dependencies[dep_name] = cleaned_dep_info
                else:
                    
                    cleaned_dependencies[dep_name] = "0.0.0"

            
            cleaned_workspace['dependencies'] = dict(sorted(cleaned_dependencies.items()))

        
        if 'package' in workspace:
            workspace_package = workspace['package']
            
            workspace_package_keys_to_keep = [
                # 'version',
                # 'homepage',
                # 'repository',
                # 'authors',
                # 'license',
                # 'keywords',
                # 'include',
                'edition',

            ]
            cleaned_workspace_package = {}
            for key in workspace_package_keys_to_keep:
                if key in workspace_package:
                    cleaned_workspace_package[key] = workspace_package[key]
            if cleaned_workspace_package:
                
                cleaned_workspace['package'] = dict(sorted(cleaned_workspace_package.items()))

        if cleaned_workspace:
            
            cleaned_workspace = dict(sorted(cleaned_workspace.items()))
            cleaned_data['workspace'] = cleaned_workspace

    
    cleaned_cargo_toml = toml.dumps(cleaned_data)

    
    cleaned_cargo_toml = '\n'.join(line.rstrip() for line in cleaned_cargo_toml.splitlines())

    return cleaned_cargo_toml


def clean_comment(cargo_toml_content):

    cargo_data = toml.loads(cargo_toml_content)
    
    
    cargo_data.pop("bin", None)
    cargo_data.pop("example", None)
    cargo_data.pop("test", None)
    cargo_data.pop("bench", None)
    cleaned_cargo_toml = toml.dumps(cargo_data)

    
    cleaned_lines = []
    for line in cleaned_cargo_toml.split('\n'):
        
        line = line.split('#', 1)[0]
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)



def get_requirements(instance: SWEbenchInstance, req_path) -> str:
    """
    Get requirements.txt for given task instance

    Args:
        instance (dict): task instance
    Returns:
        requirements.txt (str): Returns requirements.txt as string
    """
    # Attempt to find requirements.txt at each path based on task instance's repo
    commit = (
        instance["environment_setup_commit"]
        if "environment_setup_commit" in instance
        else instance["base_commit"]
    )

    return get_requirements_by_commit(instance["repo"], commit,req_path)


def get_test_directives(instance: SWEbenchInstance) -> list:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # For seq2seq code repos, testing command is fixed
    if instance["repo"] == "swe-bench/humaneval":
        return ["test.py"]

    # Get test directives from test patch and remove non-test files
    diff_pat = r"diff --git a/.* b/(.*)"
    add_file_pat = r"diff --git /dev/null b/(.*)"
    
    test_patch = instance["test_patch"]
    directives = re.findall(diff_pat, test_patch)
    new_directives = re.findall(add_file_pat, test_patch)
    directives.extend(new_directives)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]
    directives_transformed = []
    for d in directives:
        
        # if d.endswith(".rs"):
            # 
            # filename = d.split("/")[-1]  
            # 
            # filename = filename[:-3]
        directives_transformed.append(d)
    directives = directives_transformed

    return directives


def get_rust_test_command(instance: "SWEbenchInstance", specs: dict) -> list:
    
    fail_to_pass = specs.get("fail_to_pass", [])

    commands = []

    for test_case in fail_to_pass:
        if test_case.endwith(".rs"):
            
            test_file = test_case.split("/")[-1].replace(".rs", "")
            commands.append(f"cargo test --test {test_file}")
        else:
            
            commands.append(f"cargo test {test_case}")

    
    # final_command = " && ".join(commands)
    return commands

def findCrate(test_files:list):
    crates_found = set()
    # Iterate over each file path in test_files
    for file_path in test_files:
        # Check if any NON_OSDK_CRATE is in the file path
        for crate in NON_OSDK_CRATES:
            if crate in file_path:
                crates_found.add(crate)
        # Check if any OSDK_CRATE is in the file path
        for crate in OSDK_CRATES:
            if crate in file_path:
                crates_found.add(crate)
        if "test/apps" in file_path:
            crates_found.add("test/apps")
    return list(crates_found)


def str2bool(v):
    """
    Minor helper function to convert string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
