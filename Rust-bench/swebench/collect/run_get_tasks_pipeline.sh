#!/usr/bin/env bash

# Set GITHUB_TOKEN in the project root .env.

python get_tasks_pipeline.py \
    --repos '' \
    --path_prs '' \
    --path_tasks ''\
    --mode new
