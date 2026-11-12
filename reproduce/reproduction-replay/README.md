# RustBench Reproduction Replay

Replay SWE-agent trajectories in Docker and record the output of a reproduction command after each action.

## Setup

From this directory, create a Python 3.12 environment and install the dependencies:

```bash
conda create -n replay python=3.12
conda activate replay
cd replay/dependencies/SWE-agent && pip install -r requirements.txt && cd -
pip install -r requirements.txt
```

Docker must be installed and running. Container images and Rust builds require substantial disk space and time.

## Usage

Run the supplied example with `bash replay.sh`, or invoke the module directly:

```bash
python -m replay.run \
  --instance-id ast-grep__ast-grep-1707 \
  --instance-method SWE-agent \
  --instance-model claude \
  --instance-traj-path data/SWE-agent/claude/ast-grep__ast-grep-1707.traj \
  --save-dir result \
  --command "cargo test"
```

The instance ID and trajectory path identify the task and its recorded actions. The model argument labels the output; the command is executed during replay. Results are saved under `<save-dir>/<method>/<model>/<instance-id>/`, including trajectory output and a JSON report of reproduction results.

For troubleshooting, inspect the trajectory output files and ensure Docker is running, the trajectory exists, and the corresponding instance image is available.
