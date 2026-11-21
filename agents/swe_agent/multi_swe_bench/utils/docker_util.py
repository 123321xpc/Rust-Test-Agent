import logging
import os
from pathlib import Path
from typing import Optional

import docker

NETWORK_PROXY = os.environ.get("NETWORK_PROXY", "http://127.0.0.1:11321")
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY")
docker_client = docker.from_env()


def proxy_environment() -> dict[str, str]:
    values = {key: os.environ.get(key, NETWORK_PROXY) for key in PROXY_KEYS}
    values["NO_PROXY"] = os.environ.get("NO_PROXY", "localhost,127.0.0.1,::1")
    return {key: value for name, value in values.items() for key in (name, name.lower())}


def exists(image_name: str) -> bool:
    try:
        docker_client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def build(
    workdir: Path, dockerfile_name: str, image_full_name: str, logger: logging.Logger
):
    workdir = str(workdir)
    logger.info(
        f"Start building image `{image_full_name}`, working directory is `{workdir}`"
    )
    try:
        build_logs = docker_client.api.build(
            path=workdir,
            dockerfile=dockerfile_name,
            tag=image_full_name,
            rm=True,
            forcerm=True,
            decode=True,
            encoding="utf-8",
            buildargs=proxy_environment(),
        )

        for log in build_logs:
            if "stream" in log:
                logger.info(log["stream"].strip())
            elif "error" in log:
                error_message = log["error"].strip()
                logger.error(f"Docker build error: {error_message}")
                raise RuntimeError(f"Docker build failed: {error_message}")
            elif "status" in log:
                logger.info(log["status"].strip())
            elif "aux" in log:
                logger.info(log["aux"].get("ID", "").strip())

        logger.info(f"image({workdir}) build success: {image_full_name}")
    except docker.errors.BuildError as e:
        logger.error(f"build error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unknown build error occurred: {e}")
        raise e


def run(
    image_full_name: str,
    run_command: str,
    output_path: Optional[Path] = None,
    global_env: Optional[list[str]] = None,
) -> str:
    environment = proxy_environment()
    environment.update({item.split("=", 1)[0]: item.split("=", 1)[1] for item in (global_env or []) if "=" in item})
    container = docker_client.containers.run(
        image=image_full_name,
        command=run_command,
        remove=False,
        detach=True,
        stdout=True,
        stderr=True,
        environment=environment,
        privileged=True,
        network_mode="host",
    )

    output = ""
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for line in container.logs(stream=True, follow=True):
                line_decoded = line.decode("utf-8")
                f.write(line_decoded)
                output += line_decoded
    else:
        container.wait()
        output = container.logs().decode("utf-8")

    container.remove()

    return output
