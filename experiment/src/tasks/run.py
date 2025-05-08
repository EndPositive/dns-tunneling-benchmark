# fmt: off
import json
import logging
import os
import subprocess
import threading
import time
from enum import Enum
from typing import List
from urllib.parse import urlparse

import timeout_decorator
from celery import Task
from celery.states import FAILURE, SUCCESS
from python_on_whales import DockerClient

from ..lib.docker import DoubleDockerController, new_docker_controller, new_local_dns_resolvers_docker_controller
from ..lib.lib import Env, Tunnel
from ..lib.log import log_stream, log_stream_logs, logger, safe_log
from ..lib.tests import Test
from ..lib.worker import app


class NetEmClient(str, Enum):
    university = "university"
    mobile = "mobile"

    @property
    def config(self):
        university = {"delay": 29, "jitter": 0.1, "loss": 0, "rate": 1000}
        mobile = {"delay": 38, "jitter": 22, "loss": 1.0, "rate": 100}

        return {
            "university": university,
            "mobile": mobile,
        }[self.value]

class NetEmServer(str, Enum):
    national = "national"
    international = "international"

    @property
    def config(self):
        national = {"delay": 20, "jitter": 0.1, "loss": 0, "rate": 1000}
        international = {"delay": 62, "jitter": 0.5, "loss": 0.1, "rate": 1000}

        return {
            "national": national,
            "international": international,
        }[self.value]


def _get_server_ip(docker_controller: DoubleDockerController, server_docker_host: str, test: Test):
    if not docker_controller.local:
        return urlparse(server_docker_host).hostname

    if test == Test.latency:
        target = "speedtest"
    elif test == Test.file_download or test == Test.file_upload:
        target = "file-transfer"
    elif test == Test.browsing:
        target = "websites"
    else:
        target = "iperf3"

    server_container = docker_controller.server.container.inspect(f"{target}-server")
    return server_container.network_settings.networks["benchmark_experiment"].ip_address

def speedtest(docker_controller: DoubleDockerController, speedtest_server_ip: str, tunnel: str):
    # echo '[{"id": 1,"name": "LibreSpeed","server": "http://127.0.0.1:5201/backend/","pingURL": "empty.php","getIpURL": "getIP.php"}]' | librespeed-cli --no-icmp --no-download --no-upload --telemetry-level=disabled --json --server 1 --local-json -
    # echo '[{"id": 1,"name": "LibreSpeed","server": "http://172.22.0.18:5201/backend/","pingURL": "empty.php","getIpURL": "getIP.php"}]' | librespeed-cli --interface=dns0  --no-icmp --no-download --no-upload --telemetry-level=disabled --json --server 1 --local-json -
    sh = (
        "librespeed-cli "
        "--no-icmp "
        "--no-download "
        "--no-upload "
        "--telemetry-level=disabled "
        "--json "
        "--server 1 "
        "--local-json - "
    )

    if tunnel.startswith("tun/"):
        server_ip = speedtest_server_ip
        if tunnel == Tunnel.iodine:
            interface_tun_name = "dns0"
        else:
            interface_tun_name = "tun0"

        sh += f"--interface {interface_tun_name} "
    elif tunnel == Tunnel.raw:
        server_ip = speedtest_server_ip
    else:
        server_ip = "127.0.0.1"

    server_json = [
        {
            "id": 1,
            "name": "LibreSpeed",
            "server": f"http://{server_ip}:5201/backend/",
            "dlURL": "garbage.php",
            "ulURL": "empty.php",
            "pingURL": "empty.php",
            "getIpURL": "getIP.php"
        },
    ]
    print(server_json)

    json_result_client = docker_controller.client.container.execute(
        "speedtest-cli",
        ["sh", "-c", "echo '" + json.dumps(server_json) + "' | " + sh],
    )

    print(json_result_client)

    ping = json.loads(json_result_client)[0]["ping"]
    jitter = json.loads(json_result_client)[0]["jitter"]

    return ping, jitter

def get_unbound_stats(dns_resolver_ips: List[str]) -> List[dict]:
    stats = []

    for dns_resolver_ip in dns_resolver_ips:
        output = os.popen(f"unbound-control -s {dns_resolver_ip} stats_noreset").read()

        total_num_queries = -1
        total_num_queries_ip_ratelimited = -1
        for line in output.splitlines():
            if line.startswith("total.num.queries="):
                total_num_queries = int(line.split("=")[1])
            if line.startswith("total.num.queries_ip_ratelimited"):
                total_num_queries_ip_ratelimited = int(line.split("=")[1])

            if total_num_queries >= 0 and total_num_queries_ip_ratelimited >= 0:
                break

        if total_num_queries < 0:
            raise Exception("Failed to get total_num_queries from unbound-control")
        if total_num_queries_ip_ratelimited < 0:
            raise Exception("Failed to get total_num_queries_ip_ratelimited from unbound-control")

        stats.append(
            {
                "total_num_queries": total_num_queries,
                "total_num_queries_ip_ratelimited": total_num_queries_ip_ratelimited,
            }
        )

        safe_log(logger.info, f"{dns_resolver_ip}; total_num_queries={total_num_queries}")
        safe_log(logger.info, f"{dns_resolver_ip}; total_num_queries_ip_ratelimited={total_num_queries_ip_ratelimited}")

    return stats

def file_download_test(docker_controller: DoubleDockerController, speedtest_server_ip: str, tunnel: str, dns_resolver_ips: List[str]):
    sh = (
        "python main.py client "
    )

    interface_tun_name = "none"
    if tunnel.startswith("tun/"):
        server_ip = speedtest_server_ip
        if tunnel == Tunnel.iodine:
            interface_tun_name = "dns0"
        else:
            interface_tun_name = "tun0"

    elif tunnel == Tunnel.raw:
        server_ip = speedtest_server_ip
    else:
        server_ip = "127.0.0.1"

    sh += f"{interface_tun_name} {server_ip}:5201"

    # start the ncat server
    docker_controller.server.container.execute(
        "file-transfer-server",
        ["sh", "-c", "ncat -l -k -v 5201 --send-only --sh-exec 'dd bs=1MB count=10 if=/dev/urandom'"],
        detach=True,
    )

    output_stream = docker_controller.client.container.execute(
        "file-transfer-client",
        ["sh", "-c", sh],
        stream=True,
    )

    lines = []
    for _, stream_content in output_stream:
        message = stream_content.decode("utf-8").strip()
        for line in message.splitlines():
            line = line.strip()
            if not line:
                continue
            safe_log(logger.info, line, stacklevel=3)
            lines.append(line)

    unbound_stats = get_unbound_stats(dns_resolver_ips)

    epochs = []
    for line in lines:
        ns, event, error, count = line.split(",")
        epochs.append((
            int(ns),
            event,
            error,
            int(count)
        ))

    for i, unbound_stat in enumerate(unbound_stats):
        for metric_name, metric_value in unbound_stat.items():
            epochs += [
                (
                    i, metric_name, None, metric_value,
                )
            ]

    return epochs

def file_upload_test(docker_controller: DoubleDockerController, speedtest_server_ip: str, tunnel: str, dns_resolver_ips: List[str]):
    sh = (
        "python main.py server "
    )

    interface_tun_name = "none"
    if tunnel.startswith("tun/"):
        server_ip = speedtest_server_ip
        if tunnel == Tunnel.iodine:
            interface_tun_name = "dns0"
        else:
            interface_tun_name = "tun0"
    elif tunnel == Tunnel.raw:
        server_ip = speedtest_server_ip
    else:
        server_ip = "127.0.0.1"

    sh += f"none 0.0.0.0:5201"

    socat_sh = f"sleep 1 & dd if=/dev/urandom bs=1MB count=10 | socat - TCP4:{server_ip}:5201{f",bind=if:{interface_tun_name}" if interface_tun_name != 'none' else ''}"
    print(socat_sh)

    # first start the ncat server
    docker_controller.client.container.execute(
        "file-transfer-client",
        ["sh", "-c", socat_sh],
        detach=True,
    )

    output_stream = docker_controller.server.container.execute(
        "file-transfer-server",
        ["sh", "-c", sh],
        stream=True,
    )

    lines = []
    for _, stream_content in output_stream:
        message = stream_content.decode("utf-8").strip()
        for line in message.splitlines():
            line = line.strip()
            if not line:
                continue
            safe_log(logger.info, line, stacklevel=3)
            lines.append(line)

    unbound_stats = get_unbound_stats(dns_resolver_ips)

    epochs = []
    for line in lines:
        ns, event, error, count = line.split(",")
        epochs.append((
            int(ns),
            event,
            error,
            int(count)
        ))

    for i, unbound_stat in enumerate(unbound_stats):
        for metric_name, metric_value in unbound_stat.items():
            epochs += [
                (
                    i, metric_name, None, metric_value,
                )
            ]

    return epochs

def browsing_test(docker_controller: DoubleDockerController, speedtest_server_ip: str, website_path, tunnel: str, dns_resolver_ips: List[str]):
    if tunnel.startswith("tun/"):
        raise Exception("tun is not supported for the browsing test")
    elif tunnel == Tunnel.raw:
        server_ip = speedtest_server_ip
    else:
        server_ip = "127.0.0.1"

    output_stream = docker_controller.client.container.execute(
        "browsing-client",
        ["k6", "run", "script.js"],
        stream=True,
        envs={
            "BASE_URL": f"http://{server_ip}:5201/{website_path}",
        }
    )

    for _, stream_content in output_stream:
        message = stream_content.decode("utf-8").strip()
        for line in message.splitlines():
            safe_log(logger.info, line, stacklevel=3)

    unbound_stats = get_unbound_stats(dns_resolver_ips)

    summary_json = docker_controller.client.container.execute(
        "browsing-client",
        ["cat", "summary.json"]
    )

    iteration_duration = json.loads(summary_json)["metrics"]["iteration_duration"]["values"]
    avg = iteration_duration["avg"]
    min = iteration_duration["min"]
    med = iteration_duration["med"]
    max = iteration_duration["max"]
    p90 = iteration_duration["p(90)"]
    p95 = iteration_duration["p(95)"]
    test_duration = json.loads(summary_json)["state"]["testRunDurationMs"]

    stats = []
    for i, stat in enumerate(unbound_stats):
        for metric_name, metric_value in stat.items():
            stats += [
                (
                    test_duration, i, metric_name, metric_value,
                )
            ]

    return avg, min, med, max, p90, p95, stats

@timeout_decorator.timeout(60)
def _execute_iperf3(docker_controller: DoubleDockerController, sh: str):
    log_stream(docker_controller.client.container.execute, "iperf3-client", ["sh", "-c", sh])

def iperf3(docker_controller: DoubleDockerController, iperf3_server_ip: str, tunnel: str, request_id: str,
           test: Test):
    sh = (
        f"iperf3 "
        f"--time 30 "
        f"--length 1k "
        f"--set-mss 1460 "
        f"--bitrate 50m "
        f"--congestion cubic "
    )

    interface_tun_name = ""
    if tunnel.startswith("tun/") or tunnel == Tunnel.raw:
        if tunnel == Tunnel.iodine:
            interface_tun_name = "dns0"
        elif tunnel == Tunnel.tuns:
            interface_tun_name = "tun0"

        if interface_tun_name:
            sh += f"--bind-dev {interface_tun_name} "

        sh += f"--client {iperf3_server_ip} "
    else:
        sh += (
            f"--client 127.0.0.1 "
        )

    if test == Test.bidir:
        sh += "--bidir "
    elif test == test.download:
        sh += "--reverse "
    elif test == test.upload:
        # upload is implied default
        pass
    else:
        raise Exception(f"Not supported test for iperf3 recording: {test}")

    logger.info(f"Running iperf3 with the following command: {sh}")

    with NetworkCapture(docker_controller, interface_tun_name, request_id, tunnel) as capture:
        safe_log(logger.info, "Starting test")
        try:
            _execute_iperf3(docker_controller, sh)
        except timeout_decorator.TimeoutError:
            safe_log(logger.error, "iperf3 execution timed out")
        log_stream(docker_controller.client.compose.logs, "iperf3-client", no_log_prefix=True)

    log_stream(docker_controller.server.compose.logs, "iperf3-server", no_log_prefix=True)

    return capture.syn_ack_epoch


@app.task(bind=True)
def run(
        self: Task, connection, tunnel: Tunnel, net_em_client: NetEmClient, net_em_server: NetEmServer, dns_resolvers: List[str], dns_resolver_rate_limit: int, verbose: bool, test: Test, website_path: str,
        client_docker_host: str = None, server_docker_host: str = None, dry_run: bool = False
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    local = client_docker_host == server_docker_host

    if not local:
        safe_log(logger.info, "Ignoring local network emulation settings in remote context")

    safe_log(logger.info, f"Downing entire compose project")
    downer = DockerClient(compose_files=["docker-compose-common.yaml"])
    log_stream_logs(downer.compose.down, timeout=1, remove_orphans=True)

    dns_resolvers_ips = [ip for ip in dns_resolvers if ip != "local"]

    safe_log(logger.info, "Bringing up local DNS resolvers")
    local_dns_resolvers_docker_controller = new_local_dns_resolvers_docker_controller(dns_resolvers)
    if local_dns_resolvers_docker_controller:
        log_stream_logs(local_dns_resolvers_docker_controller.compose.up, services=["dns-resolver"], wait=True)
        local_dns_resolvers_containers = local_dns_resolvers_docker_controller.compose.ps(services=["dns-resolver"])
        local_dns_resolvers_ips = [container.network_settings.networks["benchmark_experiment"].ip_address for container in local_dns_resolvers_containers]
        dns_resolvers_ips.extend(local_dns_resolvers_ips)

    env = Env(dns_resolvers_ips, local=local, server_host=urlparse(server_docker_host).hostname if server_docker_host else server_docker_host, test=test)
    double_docker_controller = new_docker_controller(env, tunnel, client_host=client_docker_host, server_host=server_docker_host)

    safe_log(logger.info, "Bringing up dummy containers")
    log_stream_logs(double_docker_controller.client.compose.up, wait=True, services=["dummy-client"])
    log_stream_logs(double_docker_controller.server.compose.up, wait=True, services=["dummy-server"])

    if local:
        # add network emulation to the client and server
        docker_veth_names = get_docker_veth_names()
        apply_net_em(docker_veth_names["dummy-client"], net_em=net_em_client.config)
        if tunnel != Tunnel.raw:
            apply_net_em(docker_veth_names["dummy-server"], net_em=net_em_server.config)

    safe_log(logger.info, "Bringing up new containers")
    log_stream_logs(double_docker_controller.server.compose.up, wait=True)
    log_stream_logs(double_docker_controller.client.compose.up, wait=True)

    if local and tunnel == Tunnel.raw:
        docker_veth_names = get_docker_veth_names()
        apply_net_em(docker_veth_names["iperf3-server"], net_em=net_em_server.config)
        apply_net_em(docker_veth_names["speedtest-server"], net_em=net_em_server.config)
        apply_net_em(docker_veth_names["websites-server"], net_em=net_em_server.config)

    if dry_run:
        self.update_state(state=SUCCESS)
        return

    sync_task_metadata(connection, self.request.id, tunnel, dns_resolvers, dns_resolver_rate_limit, test, net_em_client, net_em_server, website_path)
    connection.commit()

    safe_log(logger.info, "Waiting for all components to initialize (10 seconds)")
    time.sleep(10)

    if tunnel == Tunnel.iodine:
        safe_log(logger.info, "Waiting some extra for iodine to initialize (10 seconds)")
        time.sleep(10)

    try:
        server_ip = _get_server_ip(double_docker_controller, server_docker_host, test)

        if test == Test.latency:
            latency, jitter = speedtest(double_docker_controller, server_ip, tunnel)
            sync_speedtest_result(connection, self.request.id, latency, jitter)
        elif test == Test.file_download:
            epochs = file_download_test(double_docker_controller, server_ip, tunnel, dns_resolvers_ips)
            sync_file_transfer_results(connection, self.request.id, epochs)
        elif test == Test.file_upload:
            epochs = file_upload_test(double_docker_controller, server_ip, tunnel, dns_resolvers_ips)
            sync_file_transfer_results(connection, self.request.id, epochs)
        elif test == Test.browsing:
            avg, min, med, max, p90, p95, unbound_stats = browsing_test(double_docker_controller, server_ip, website_path, tunnel, dns_resolvers_ips)
            sync_unbound_metrics(connection, self.request.id, unbound_stats)
            sync_browsing_test_results(connection, self.request.id, avg, min, med, max, p90, p95)
        else:
            syn_ack_epoch = iperf3(double_docker_controller, server_ip, tunnel, self.request.id, test)
            sync_iperf3_results(connection, self.request.id, syn_ack_epoch)

        connection.commit()

        self.update_state(state=SUCCESS)

        return
    except Exception as e:
        connection.rollback()
        self.update_state(state=FAILURE, meta={"error": str(e)})

        # log_stream(double_docker_controller.client.compose.logs)
        # log_stream(double_docker_controller.server.compose.logs)
        raise e


def container_pid(controller: DockerClient, container_name: str) -> int:
    return controller.container.inspect(container_name).state.pid


def run_dumpcap(client: DockerClient, container: str, interface_tun_name: str, file_name: str):
    safe_log(logger.info, f"Running dumpcap in {container}")

    sh = [
        "dumpcap",
        "-w", f"/{file_name}.pcap",
        "-B", "8192",
        "-q",
        "-i", "eth0",
        "-i", "lo",
    ]

    if interface_tun_name:
        sh.extend(["-i", interface_tun_name])

    try:
        log_stream(client.container.execute,
                   f"{container}", sh)
    except Exception as e:
        logger.error(f"dumpcap process failed: {e}")


class NetworkCapture:
    def __init__(self, docker_controller: DoubleDockerController, interface_tun_name, file_name, tunnel):
        self.docker_controller = docker_controller
        self.interface_tun_name = interface_tun_name
        self.file_name = file_name

        self.client_dumpcap = "dumpcap-client"
        self.server_dumpcap = "iperf3-dumpcap-server" if tunnel == Tunnel.raw else "dumpcap-server"

    def __enter__(self):
        # Start the dumpcap in a background thread
        self.dumpcap_thread_client = threading.Thread(target=run_dumpcap, args=(self.docker_controller.client, self.client_dumpcap, self.interface_tun_name, self.file_name))
        self.dumpcap_thread_client.start()

        self.dumpcap_thread_server = threading.Thread(target=run_dumpcap, args=(self.docker_controller.server, self.server_dumpcap, self.interface_tun_name, self.file_name))
        self.dumpcap_thread_server.start()

        safe_log(logger.info, "Waiting for packet capture to initialize (10 seconds)")
        time.sleep(10)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        file_name = self.file_name


        safe_log(logger.info, "Waiting for wild packets to die (10 seconds)")
        time.sleep(10)

        safe_log(logger.info, "Stopping tshark process")

        log_stream(self.docker_controller.client.container.execute, self.client_dumpcap, ["killall", "dumpcap"])
        log_stream(self.docker_controller.server.container.execute, self.server_dumpcap, ["killall", "dumpcap"])

        # Wait for the thread to join, ensuring that `dumpcap` has finished
        self.dumpcap_thread_client.join(timeout=10)
        self.dumpcap_thread_server.join(timeout=10)

        self.syn_ack_epoch = self.docker_controller.client.container.execute(self.client_dumpcap, ["tshark", "-r", f"/{file_name}.pcap", "-2R", "tcp.flags.syn==1 && tcp.flags.ack==1", "-c", "1", "-T", "fields", "-e", "frame.time_epoch"])

        sh = (
            f"tshark -r /{file_name}.pcap -T fields "
            f"-e frame.time_epoch "
            f"-e tcp.len "
            f"-e tcp.srcport "
            f"-e tcp.dstport "
            f"-e udp.length "
            f"-e udp.srcport "
            f"-e udp.dstport "
            f"-E separator=, "
            f"-E quote=d "
            f"-E header=y > /{file_name}.csv"
        )

        safe_log(logger.info, "Running tshark to convert pcap to csv: sh -c '" + sh + "'")
        log_stream(self.docker_controller.client.container.execute, self.client_dumpcap, ["sh", "-c", sh])
        log_stream(self.docker_controller.server.container.execute, self.server_dumpcap, ["sh", "-c", sh])

        self.docker_controller.client.container.copy(source=(self.client_dumpcap, f"/{file_name}.pcap"), destination=f"./celery/artifacts/{file_name}-client.pcap")
        self.docker_controller.client.container.copy(source=(self.client_dumpcap, f"/{file_name}.csv"), destination=f"./celery/artifacts/{file_name}-client.csv")
        self.docker_controller.server.container.copy(source=(self.server_dumpcap, f"/{file_name}.pcap"), destination=f"./celery/artifacts/{file_name}-server.pcap")
        self.docker_controller.server.container.copy(source=(self.server_dumpcap, f"/{file_name}.csv"), destination=f"./celery/artifacts/{file_name}-server.csv")


def get_docker_veth_names() -> dict[str, str]:
    veth_names = {}
    docker_veth_cmd = "sudo sh tools/docker-veth/docker-veth.sh"
    safe_log(logger.info, f"Running docker-veth: {docker_veth_cmd}")
    output = subprocess.run(docker_veth_cmd.split(), check=True, capture_output=True).stdout.decode("utf-8")
    lines = output.splitlines()
    for line in lines:
        cols = line.split()
        if len(cols) != 3:
            continue
        vethif, _, container_name = line.split()
        veth = vethif.split("@")[0]
        veth_names[container_name] = veth

    return veth_names


def apply_net_em(veth: str, net_em: dict):
    if not net_em:
        return

    delay = net_em["delay"]
    jitter = net_em["jitter"]
    loss = net_em["loss"]
    rate = net_em["rate"]

    netem_cmd = f"sudo tc qdisc add dev {veth} root netem delay {delay}ms {jitter}ms loss {loss}% rate {rate}mbit limit 100000"
    safe_log(logger.info, f"Applying network emulation on {veth}: {netem_cmd}")
    subprocess.run(netem_cmd.split(), check=True)

    safe_log(logger.info, f"Network emulation applied successfully on {veth}")


def sync_task_metadata(connection, task_id: str, tunnel: Tunnel, dns_resolvers: List[str], dns_resolver_rate_limit: int, test: Test, net_em_client: NetEmClient, net_em_server: NetEmServer, website_path: str):
    with connection.cursor() as cursor:
        # Insert metadata into task_metadata table
        cursor.execute("""
                INSERT INTO task_metadata (task_id, tunnel_name, dns_resolvers, dns_resolver_rate_limit, test_type, netem_client_name, netem_client, netem_server_name, netem_server, website_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (task_id, tunnel.value, json.dumps(dns_resolvers), dns_resolver_rate_limit, test.value, net_em_client.value, json.dumps(net_em_client.config), net_em_server.value, json.dumps(net_em_server.config), website_path))

def sync_unbound_metrics(connection, task_id, metrics: List[tuple]):
    with connection.cursor() as cursor:
        _to_insert = [(task_id,) + metric for metric in metrics]
        print(_to_insert)
        cursor.executemany("""
                INSERT INTO unbound_metrics (task_id, time, resolver_id, metric, value)
                VALUES (%s, %s, %s, %s, %s)
            """, (_to_insert))

def sync_browsing_test_results(connection, task_id: str, avg: int, min: int, med: int, max: int, p90: int, p95: int):
    with connection.cursor() as cursor:
        cursor.execute("""
                INSERT INTO browsing_test_results (task_id, avg, min, med, max, p90, p95)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (task_id, avg, min, med, max, p90, p95))


def sync_speedtest_result(connection, task_id: str, latency: int, jitter: int):
    with connection.cursor() as cursor:
        cursor.execute("""
                INSERT INTO speedtest_result (task_id, latency, jitter)
                VALUES (%s, %s, %s)
            """, (task_id, latency, jitter))

def sync_iperf3_results(connection, task_id: str, syn_ack_epoch: float):
    with connection.cursor() as cursor:
        for side in ["client", "server"]:
            # Load packet data from CSV into packet_data table (without task_id initially)
            csv_file_path = f"./celery/artifacts/{task_id}-{side}.csv"
            with open(csv_file_path, 'r') as f:
                cursor.copy_expert("""
                    COPY packet_data (frame_time, tcp_len, tcp_srcport, tcp_dstport, udp_length, udp_srcport, udp_dstport)
                    FROM STDIN WITH CSV HEADER
                """, f)

            # Update the client column
            # Update packet_data to include the task_id
            # Adjust frame_time in packet_data
            cursor.execute(f"""
                UPDATE packet_data
                SET client = %s, task_id = %s, frame_time = frame_time - %s + 10
                WHERE task_id IS NULL;
            """, (side == "client", task_id, syn_ack_epoch))

def sync_file_transfer_results(connection, task_id, epochs):
    with connection.cursor() as cursor:
        _to_insert = [(task_id,) + epoch for epoch in epochs]
        cursor.executemany("""
            INSERT INTO file_download_result (task_id, time, event_type, error, bytes_received)
            VALUES (%s, %s, %s, %s, %s)
        """, _to_insert)
