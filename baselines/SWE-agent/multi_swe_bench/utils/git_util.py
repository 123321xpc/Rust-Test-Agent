import logging
import os
import subprocess
from pathlib import Path

from git import GitError, Repo


def clone_repository(destination: Path, org: str, repo: str):
    repo_url = f"https://github.com/{org}/{repo}.git"
    repo_path = destination.absolute() / repo
    proxy = os.environ.get("NETWORK_PROXY", "http://127.0.0.1:11321")
    proxy_env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY"):
        proxy_env.setdefault(key, proxy)
        proxy_env.setdefault(key.lower(), proxy)
    proxy_env.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    proxy_env.setdefault("no_proxy", proxy_env["NO_PROXY"])
    subprocess.run(
        ["git", "clone", repo_url, str(repo_path)], env=proxy_env,
        check=True,
    )


def get_all_commit_hashes(repo_path: Path, logger: logging.Logger) -> set[str]:
    try:
        repo = Repo(repo_path)

        commit_hashes = set()
        for commit in repo.iter_commits("--all"):
            commit_hashes.add(commit.hexsha)

        return commit_hashes
    except GitError as e:
        logger.error(f"Git error occurred: {e}")
        return set()

