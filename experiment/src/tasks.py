import logging
import time
from urllib.parse import urlparse

from celery.app.task import Task
from celery.states import FAILURE, SUCCESS

from .lib.lib import DoubleDockerController, Env, Tunnel, log_stream, log_stream_logs, logger, \
    new_docker_controller, safe_log, network_capture
from .lib.worker import app

def _build(docker_controller: DoubleDockerController):
    for docker_controller in [docker_controller.client, docker_controller.server]:
        safe_log(logger.info, "Building containers")
        log_stream_logs(docker_controller.compose.build, progress="plain")

def _down_up(double_docker_controller: DoubleDockerController):
    for docker_controller in [double_docker_controller.client, double_docker_controller.server]:
        safe_log(logger.info, "Downing any running containers")
        log_stream_logs(docker_controller.compose.down, timeout=1, remove_orphans=True)

    for docker_controller in [double_docker_controller.client, double_docker_controller.server]:
        safe_log(logger.info, "Bringing up new containers")
        log_stream_logs(docker_controller.compose.up, wait=True)

    safe_log(logger.info, "Waiting for all components to initialize (10 seconds)")
    time.sleep(10)

def _get_iperf3_server_ip(docker_controller: DoubleDockerController, server_docker_host: str):
    if not docker_controller.local:
        return urlparse(server_docker_host).hostname

    iperf3_server_container = docker_controller.server.container.inspect("iperf3-server")
    return iperf3_server_container.network_settings.networks["benchmark_experiment"].ip_address

def _record_iperf3(docker_controller: DoubleDockerController, iperf3_server_ip: str, tunnel: str, request_id: str, upload: bool, download: bool):
    sh = (
        f"iperf3 "
        f"--time 90 "
        f"--length 1k "
        f"--set-mss 1460 "
    )

    interface_tun_name = ""
    if tunnel.startswith("tun/"):
        if tunnel == Tunnel.iodine:
            interface_tun_name = "dns0"
        else:
            interface_tun_name = "tun0"

    if tunnel.startswith("tun/"):
        sh += (
            f"--bind-dev {interface_tun_name} "
        )

    if tunnel.startswith("socks/"):
        sh += (
            f"--client 127.0.0.1 "
        )
    else:
        sh += (
            f"--client {iperf3_server_ip} "
        )

    if upload and download:
        sh += "--bidir "
    elif download:
        sh += "--reverse "
    # upload is implied default

    logger.info(f"Running iperf3 with the following command: {sh}")

    file_name = f"/celery/artifacts/{request_id}"

    with network_capture(docker_controller, interface_tun_name, file_name):
        safe_log(logger.info, "Starting test")
        log_stream(
            docker_controller.client.container.execute,
            "iperf3-client",
            ["sh", "-c", sh],
        )

@app.task(bind=True)
def run(
        self: Task, tunnel: str, use_dns_resolver: bool, verbose: bool, download=True, upload=True, client_docker_host: str = None, server_docker_host: str = None
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    env = Env(use_dns_resolver, raw=tunnel == Tunnel.raw, local=client_docker_host == server_docker_host, server_host=urlparse(server_docker_host).hostname if server_docker_host else server_docker_host)
    docker_controller = new_docker_controller(env, tunnel, use_dns_resolver, client_host=client_docker_host, server_host=server_docker_host)

    _down_up(docker_controller)

    try:
        iperf3_server_ip = _get_iperf3_server_ip(docker_controller, server_docker_host)

        _record_iperf3(docker_controller, iperf3_server_ip, tunnel, self.request.id, upload, download)

        self.update_state(
            state=SUCCESS,
            meta={
                "tunnel": tunnel,
                "use_dns_resolver": use_dns_resolver,
                "upload": upload,
                "download": download,
                "local": docker_controller.local,
            }
        )

        return
    except Exception as e:
        self.update_state(
            state=FAILURE,
            meta={
                "tunnel": tunnel,
                "use_dns_resolver": use_dns_resolver,
                "upload": upload,
                "download": download,
                "local": docker_controller.local,
            }
        )

        log_stream(docker_controller.client.compose.logs)
        log_stream(docker_controller.server.compose.logs)
        raise e
    finally:
        log_stream_logs(docker_controller.client.compose.down, volumes=True, timeout=1)
        log_stream_logs(docker_controller.server.compose.down, volumes=True, timeout=1)


@app.task(bind=True)
def restart(
        self: Task, tunnel: str, use_dns_resolver: bool, verbose: bool, client_docker_host: str = None, server_docker_host: str = None
):
    env = Env(use_dns_resolver, raw=tunnel == Tunnel.raw, local=client_docker_host == server_docker_host, server_host=urlparse(server_docker_host).hostname if server_docker_host else server_docker_host)
    docker_controller = new_docker_controller(env, tunnel, use_dns_resolver, client_host=client_docker_host, server_host=server_docker_host)

    _down_up(docker_controller)

@app.task(bind=True)
def build(
        self: Task, tunnel: str, use_dns_resolver: bool, verbose: bool, client_docker_host: str = None, server_docker_host: str = None
):
    env = Env(use_dns_resolver, raw=tunnel == Tunnel.raw, local=client_docker_host == server_docker_host, server_host=urlparse(server_docker_host).hostname if server_docker_host else server_docker_host)
    docker_controller = new_docker_controller(env, tunnel, use_dns_resolver, client_host=client_docker_host, server_host=server_docker_host)

    _build(docker_controller)


@app.task(bind=True)
def destroy(
        self: Task,
        client_docker_host: str = None,
        server_docker_host: str = None,
):
    env = Env(True, raw=False, local=client_docker_host == server_docker_host, server_host=urlparse(server_docker_host).hostname if server_docker_host else server_docker_host)
    docker_controller = new_docker_controller(env, Tunnel.raw.value, use_dns_resolver=False, client_host=client_docker_host, server_host=server_docker_host)
    log_stream_logs(docker_controller.client.compose.down, volumes=True, timeout=1, remove_orphans=True)
    log_stream_logs(docker_controller.server.compose.down, volumes=True, timeout=1, remove_orphans=True)
