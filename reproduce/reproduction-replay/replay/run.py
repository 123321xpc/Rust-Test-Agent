"""
Main execution script for replaying and analyzing Software Engineering (SWE) agent trajectories.

This module provides functionality to replay trajectories from different SWE agents
(SWE-agent) and analyze their output changes when executing
specific commands. It's primarily used for studying agent behavior and performance comparison.
"""

import json
import os

import typer
from typing import List
from typing import Optional
from typing import Tuple


from replay.instance import make_mswe_instance
from replay.instance import SWEAgentMultiSWEInstance
from replay.instance import MultiSWEInstance
from replay.instance import MultiSWEInstanceMethod
from replay.utils.log import log_open_file_logger
from replay.utils.sweagent import sweagent_function_response_extract_status, sweagent_internal_unedit_commands
from replay.utils.sweagent import sweagent_replay
from replay.utils.sweagent import sweagent_function_call_extract_command
from replay.utils.sweagent import sweagent_trajs_add_command
from replay.utils.text import text_remove_cargo_warning
from replay.utils.uuid import uuid_str
from replay.utils.error import error_message


app = typer.Typer()

# Global environment setup commands for consistent container initialization
# These commands configure Rust environment and proxy settings for reproducibility
GLOBAL_ENVIRONMENT_SETUP_COMMANDS = [
    f'export RUSTFLAGS="-Awarnings"',  # Suppress Rust compiler warnings
    f'export http_proxy="{os.getenv("http_proxy") or ""}"',  # Use system HTTP proxy if available
    f'export https_proxy="{os.getenv("https_proxy") or ""}"',  # Use system HTTPS proxy if available
    f'alias cargo="cargo --quiet"',  # Make cargo commands less verbose
]


def _wrap_output(
    out: str,
) -> str:
    """
    Clean up command output by removing Cargo warnings.

    Args:
        out: Raw command output string

    Returns:
        Cleaned output string with Cargo warnings removed
    """
    out = text_remove_cargo_warning(out)
    return out


def _wrap_exitcode_output(
    ex_out: Tuple[int, str],
) -> Tuple[int, str]:
    """
    Clean up command output with exit code by removing Cargo warnings.

    Args:
        ex_out: Tuple containing (exit_code, output_string)

    Returns:
        Tuple with cleaned output string
    """
    ex, out = ex_out
    out = text_remove_cargo_warning(out)
    return ex, out


def sweagent_prepare_trajs(
    instance: MultiSWEInstance,
    sweagent_replay_find_target_round_trajs: List[dict],
    sweagent_home_dir: str,
    command: str,
):
    """
    Prepare SWEAgent trajectory by adding environment setup and command execution steps.

    This function constructs a replay trajectory by:
    1. Adding global environment setup commands
    2. Configuring git settings
    3. Processing each LLM request/response pair to extract commands
    4. Adding test command execution after each valid agent action
    5. Recovering working directory after each step

    Args:
        instance: Multi-SWE instance containing trajectory data
        sweagent_replay_find_target_round_trajs: List to populate with trajectory steps
        sweagent_home_dir: Home directory path for the agent
        command: Test command to execute after each step
    """
    # Add global environment setup commands
    for cmd in GLOBAL_ENVIRONMENT_SETUP_COMMANDS:
        sweagent_trajs_add_command(
            sweagent_replay_trajs=sweagent_replay_find_target_round_trajs,
            message=_sweagent_thought_messages["setup_environment"],
            command=cmd,
        )
    # Configure git settings and add git alias for cleaner output
    sweagent_trajs_add_command(
        sweagent_replay_trajs=sweagent_replay_find_target_round_trajs,
        message=_sweagent_thought_messages["setup_environment"],
        command=f'cd {sweagent_home_dir} && git config --global user.name "msweagent" && git config --global user.email "msweagent@common.dev" && alias git="git --no-pager"',
    )
    # Process each LLM interaction pair
    for llm_request_traj, llm_response_traj in zip(instance.get_llm_call_request_trajs(), instance.get_llm_call_response_trajs()):
        # Extract command from the request trajectory
        sweagent_command = sweagent_function_call_extract_command(llm_request_traj)
        if not sweagent_command:
            continue
        sweagent_replay_find_target_round_trajs.append(llm_request_traj)
        # Skip internal commands that don't require test execution
        if any(sweagent_command.startswith(command) for command in sweagent_internal_unedit_commands()):
            continue
        # Add test command execution
        sweagent_trajs_add_command(
            sweagent_replay_trajs=sweagent_replay_find_target_round_trajs,
            message=_sweagent_thought_messages["run_command"],
            command=command,
        )
        try:
            # Extract working directory from response and add command to recover it
            _, working_dir = sweagent_function_response_extract_status(llm_response_traj)
            sweagent_trajs_add_command(
                sweagent_replay_trajs=sweagent_replay_find_target_round_trajs,
                message=_sweagent_thought_messages["recover_working_dir"],
                command=f"cd '{working_dir}'",
            )
        except:
            # Ignore errors in working directory extraction
            pass


def sweagent_run_replay(
    instance: MultiSWEInstance,
    instance_playback_analysis_dir_path: str,
    command: Optional[str] = None,
) -> None:
    """
    Execute SWEAgent trajectory replay and analyze command execution results.

    This function performs a three-step process:
    1. Prepare trajectory with environment setup and test commands
    2. Execute the replay using SWEAgent
    3. Parse results and extract command execution observations

    Args:
        instance: Multi-SWE instance containing trajectory data
        instance_playback_analysis_dir_path: Directory to save analysis results
        command: Optional test command to execute after each step
    """

    # Define output file paths
    instance_playback_analysis_traj_sweagent_output_path = os.path.join(
        instance_playback_analysis_dir_path,
        f"{instance.id}__sweagent_output.traj",
    )
    instance_playback_analysis_traj_sweagent_result_path = os.path.join(
        instance_playback_analysis_dir_path,
        f"{instance.id}__sweagent_result.traj",
    )
    instance_playback_analysis_result_path = os.path.join(
        instance_playback_analysis_dir_path,
        f"{instance.id}.json",
    )

    # Skip if results already exist
    if os.path.exists(instance_playback_analysis_result_path):
        print(f"skip {instance} because result already exists")
        return

    # Set up SWEAgent home directory
    sweagent_home_dir = f"/home/{instance.repo}"

    try:
        # Step 1: Prepare SWEAgent trajectory with commands
        sweagent_replay_find_target_round_trajs: List[dict] = []
        sweagent_prepare_trajs(
            instance=instance,
            sweagent_replay_find_target_round_trajs=sweagent_replay_find_target_round_trajs,
            sweagent_home_dir=sweagent_home_dir,
            command=command,
        )

        # Step 2: Execute SWEAgent replay with prepared trajectory
        sweagent_replay_output, sweagent_replay_result, _ = sweagent_replay(
            instance=instance,
            replay_trajs=sweagent_replay_find_target_round_trajs,
            timeout=3600,  # 1 hour timeout
        )
        # Save raw replay output and results
        with open(instance_playback_analysis_traj_sweagent_output_path, "w") as f:
            f.write(sweagent_replay_output)
        with open(instance_playback_analysis_traj_sweagent_result_path, "w") as f:
            f.write(sweagent_replay_result)

        # Step 3: Parse and filter replay results
        results = []
        sweagent_replay_result = json.loads(sweagent_replay_result)
        prev_observation = None
        for round_number, traj in enumerate(sweagent_replay_result["trajectory"], start=1):
            thought: str = traj["thought"]
            observation: str = _wrap_output(traj["observation"])
            # Only process command execution steps (not setup or directory recovery)
            if not thought.startswith(_sweagent_thought_messages["run_command"]):
                continue
            # Skip duplicate observations to avoid redundancy
            if prev_observation == observation:
                continue
            prev_observation = observation
            results.append(
                {
                    "round": f"{round_number:02d}",
                    "command": command,
                    "exit_code": None,  # SWEAgent doesn't provide exit codes
                    "output": observation,
                }
            )

        # Save processed results to JSON
        with open(instance_playback_analysis_result_path, "w") as fp:
            json.dump(
                results,
                fp,
                indent=4,
            )

    except Exception as e:
        raise e


@app.command()
def main(
    instance_id: str = typer.Option(...),
    instance_method: str = typer.Option(...),
    instance_model: str = typer.Option(...),
    instance_traj_path: str = typer.Option(...),
    save_dir: str = typer.Option(...),
    command: str = typer.Option(...),
) -> None:
    """
    Main entry point for trajectory replay analysis.

    This command-line interface allows users to replay trajectories from different
    SWE agents and analyze how command execution results change over time.

    Args:
        instance_id: Unique identifier for the SWE instance
        instance_method: Agent method (SWE-agent)
        instance_model: Model name used by the agent
        instance_traj_path: Path to the trajectory file
        save_dir: Directory to save analysis results
        command: Test command to execute after each trajectory step
    """

    # Validate that the instance method is supported
    if instance_method not in [
        MultiSWEInstanceMethod.SWEAGENT,
    ]:
        print(f"instance method '{instance_method}' is not implemented")
        raise typer.Exit(code=1)

    # Create the appropriate instance based on the method
    instance = make_mswe_instance(
        instance_id=instance_id,
        instance_method=instance_method,
        instance_model=instance_model,
        instance_traj_path=instance_traj_path,
    )

    # Create output directory structure: save_dir/method/model/instance_id
    instance_playback_analysis_dir_path = os.path.join(
        save_dir,
        instance.method,
        instance.model,
        instance.id,
    )

    os.makedirs(instance_playback_analysis_dir_path, exist_ok=True)

    try:

        sweagent_run_replay(
            instance=instance,
            instance_playback_analysis_dir_path=instance_playback_analysis_dir_path,
            command=command,
        )

    except Exception as e:
        print(f"reproduction replay failed: {error_message(e)}")


if __name__ == "__main__":
    # Run the Typer CLI application
    app()
