import logging
import os
import time

from celery.app.task import Task
from celery.states import FAILURE, SUCCESS

from .lib.lib import Env, Tunnel, container_pid, log_stream, log_stream_logs, logger, \
    new_docker_controller, return_stream, start_tshark_process
from .lib.worker import app, artifact_dir


@app.task(bind=True)
def run(
        self: Task, tunnel: str, use_dns_resolver: bool, verbose: bool
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    env = Env(use_dns_resolver, raw=tunnel == Tunnel.raw)
    docker_controller = new_docker_controller(env, tunnel, use_dns_resolver, client_host=None, server_host=None)

    docker_controller.compose.build()

    logger.info("Downing any running containers")
    log_stream_logs(docker_controller.compose.down, timeout=1, remove_orphans=True)

    logger.info("Bringing up new containers")
    log_stream_logs(docker_controller.compose.up, wait=True)

    logger.info("Waiting for all components to initialize (10 seconds)")
    time.sleep(10)

    # veth_names = get_veth_names()
    # apply_netem(veth_names["socks-client"], 10, 100, 10, 5)
    # apply_netem(veth_names["iperf3-server"], 10, 100, 10, 5)

    if tunnel == Tunnel.iodine:
        interface_tun_name = "dns0"
    else:
        interface_tun_name = "tun0"

    client_container = "iperf3-client"

    pid = container_pid(docker_controller, client_container)

    try:
        tshark_file_name = os.path.join(artifact_dir, f"{self.request.id}.csv")
        with start_tshark_process(interface_tun_name, "eth0", pid, f"{tshark_file_name}"):
            shared_opts = [
                "iperf3",
                "--bind-dev", interface_tun_name,
                "--client", env.IPV4_ADDRESS_IPERF3_SERVER,
                "--time", "30",
                "--length", "1k",
                "-J",
                "--get-server-output"
            ]

            logger.info("Starting upload test")
            output = return_stream(
                docker_controller.container.execute,
                client_container,
                shared_opts,
            )
            with open(os.path.join(artifact_dir, f"{self.request.id}-upload.json"), "w") as f:
                f.write(output)

            logger.info("Starting download test")
            output = return_stream(
                docker_controller.container.execute,
                client_container,
                shared_opts + ["--reverse"],
            )
            with open(os.path.join(artifact_dir, f"{self.request.id}-download.json"), "w") as f:
                f.write(output)

            logger.info("Starting bidirectional test")
            output = return_stream(
                docker_controller.container.execute,
                client_container,
                shared_opts + ["--bidir"],
            )
            with open(os.path.join(artifact_dir, f"{self.request.id}-bidir.json"), "w") as f:
                f.write(output)

        self.update_state(
            state=SUCCESS,
            meta={
                "tunnel": tunnel,
                "use_dns_resolver": use_dns_resolver,
            }
        )

        return
    except Exception as e:
        self.update_state(
            state=FAILURE,
            meta={
                "tunnel": tunnel,
                "use_dns_resolver": use_dns_resolver,
            }
        )

        log_stream(docker_controller.compose.logs)
        raise e
    # finally:
    #     log_stream_logs(docker_controller.compose.down, volumes=True, timeout=1)


@app.task(bind=True)
def destroy(
        self: Task
):
    env = Env()
    docker_controller = new_docker_controller(env, Tunnel.raw.value, use_dns_resolver=False, client_host=None, server_host=None)
    log_stream_logs(docker_controller.compose.down, volumes=True, timeout=1, remove_orphans=True)
