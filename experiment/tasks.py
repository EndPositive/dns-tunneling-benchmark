import logging
import os
import time

from python_on_whales import DockerClient

from celery import Task
from lib import (
    SERVICES,
    create_tmp_env_file,
    env,
    get_interface_name,
    logger,
    start_tshark_process,
    stream,
    stream_logs,
)
from worker import app


@app.task(bind=True)
def run_experiment(
    self: Task, tunnel: str, use_dns_resolver: bool, file_size: str, verbose: bool
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    tmp_env_filename = create_tmp_env_file(use_dns_resolver)

    docker_client = DockerClient(
        compose_files=[
            "docker-compose.yaml",
            f"tunnels/{tunnel}/docker-compose.yaml",
        ],
        compose_env_files=[tmp_env_filename],
    )

    docker_client.compose.build()

    logger.info("Downing any running containers")
    stream_logs(docker_client.compose.down, timeout=1)

    logger.info("Bringing up new containers")
    stream_logs(docker_client.compose.up, services=SERVICES, wait=True, color=True)

    try:
        logger.info("Inspecting created network")
        network = docker_client.network.inspect("benchmark_experiment")

        logger.info("Waiting for all components to initialize (10 seconds)")
        time.sleep(10)

        artifact_dir = "./celery/artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        pcap_file_name = os.path.join(artifact_dir, f"{self.request.id}.pcap")

        with start_tshark_process(get_interface_name(network), pcap_file_name):
            stream(
                docker_client.container.execute,
                "socks-client",
                [
                    "curl",
                    "--interface",
                    "tun0",
                    f"http://{env["IPV4_ADDRESS_BIG_FILES"]}:8000/1mb.txt",
                    "-vvv",
                    "--output",
                    f"{file_size}.txt",
                ],
            )

        return {
            "status": "completed",
            "pcap_file_name": pcap_file_name,
            "tunnel": tunnel,
            "use_dns_resolver": use_dns_resolver,
            "file_size": file_size,
        }
    except Exception as e:
        stream(docker_client.compose.logs)
        raise e
    finally:
        stream_logs(docker_client.compose.down, volumes=True, timeout=1)
