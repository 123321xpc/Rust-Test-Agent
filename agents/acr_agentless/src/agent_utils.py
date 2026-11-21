import re
import json
from loguru import logger
import os

def extract_code_blocks(text: str):
    lines = text.split('\n')
    
    code_blocks = []
    current_block = []
    in_code_block = False
    
    for line in lines:
        stripped_line = line.strip()
        
        if stripped_line.startswith('```'):
            if not in_code_block:
                in_code_block = True
                current_block = []
                if len(stripped_line) > 3:
                    language_identifier = stripped_line[3:].strip()
                    if language_identifier:
                        current_block.append(language_identifier)
            else:
                in_code_block = False
                block_content = '\n'.join(current_block).strip()
                if block_content:
                    code_blocks.append(block_content)
                current_block = []
        elif in_code_block:
            current_block.append(line)
    

    cleaned_blocks = []
    for block in code_blocks:

        lines = block.split('\n', 1)
        if len(lines) > 1 and re.match(r'^(python|bash|rust|js|javascript|typescript|ts|go|java|c|cpp|csharp|cs|ruby|php|swift|kotlin|scala|perl|r|shell|powershell|sql|html|css|xml|yaml|json|markdown|md|plaintext)$', lines[0].strip()):

            cleaned_blocks.append(lines[1].strip())
        else:
            cleaned_blocks.append(block.strip())
    
    function_blocks = [block for block in cleaned_blocks if block.strip().startswith("function:")]
    return function_blocks

def init_logger(log_path: str):
    with open(log_path, 'w') as f:
        pass
    logger.remove()
    logger.add(log_path, level="INFO", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True)
    # logger.add(lambda msg: print(msg, end=""), level="INFO")


def log_info(msg: str):
    logger.info(msg)

def log_error(msg: str):
    logger.error(msg)


def _resolve_bug_report_path() -> str:
    from pathlib import Path
    env_dir = os.environ.get("RUSTBENCH_DATA_DIR")
    candidates = [
        os.path.join(env_dir, "rb_all_bug_report.json") if env_dir else None,
        "/RTAgent/data/rb_all_bug_report.json",
    ]
    for parent in Path(__file__).resolve().parents:
        candidates.append(str(parent / "data" / "rb_all_bug_report.json"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rb_all_bug_report.json")


def get_instance_info(instance_id: str):
    instance_workspace_path = '/workspace/' + re.sub(r'-\d+$', '', instance_id) + "__0.1"
    data_file_path = _resolve_bug_report_path()
    with open(data_file_path, 'r', encoding='utf-8') as f:
        bugreport_dataset = json.load(f)
        if instance_id not in bugreport_dataset:
            raise Exception(f"Instance ID {instance_id} not found in bug report dataset")
        instance_bug_report = bugreport_dataset[instance_id]
        return instance_bug_report, instance_workspace_path
