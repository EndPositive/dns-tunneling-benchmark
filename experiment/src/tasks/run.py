import json
import logging
import subprocess
import threading
import time
from enum import Enum
from typing import List
from urllib.parse import urlparse

from celery import Task
from celery.states import FAILURE, SUCCESS
from python_on_whales import DockerClient

from ..lib.docker import DoubleDockerController, new_docker_controller, new_local_dns_resolvers_docker_controller
from ..lib.lib import Env, Tunnel
from ..lib.log import log_stream, log_stream_logs, logger, safe_log
from ..lib.worker import app


class NetEm(str, Enum):
    baseline = "baseline"
    high_latency = "high_latency"
    packet_loss = "packet_loss"
    congestion = "congestion"
    jitter = "jitter"
    bandwidth_limited = "bandwidth_limited"
    none = "none"

    @property
    def config(self):
        if self.value == "none":
            return None

        baseline = {"delay": 50, "jitter": 5, "loss": 0.1, "rate": 200}
        high_latency = {**baseline, "delay": 200, "jitter": 5}
        packet_loss = {**baseline, "loss": 2}
        congestion = {**baseline, "delay": 80, "jitter": 20, "loss": 0.5, "rate": 50}
        jitter = {**baseline, "jitter": 50}
        bandwidth_limited = {**baseline, "rate": 1}

        return {
            "baseline": baseline,
            "high_latency": high_latency,
            "packet_loss": packet_loss,
            "congestion": congestion,
            "jitter": jitter,
            "bandwidth_limited": bandwidth_limited,
        }[self.value]

def _get_server_ip(docker_controller: DoubleDockerController, server_docker_host: str):
    if not docker_controller.local:
        return urlparse(server_docker_host).hostname

    iperf3_server_container = docker_controller.server.container.inspect("iperf3-server")
    return iperf3_server_container.network_settings.networks["benchmark_experiment"].ip_address


class Test(str, Enum):
    upload = "upload"
    download = "download"
    bidir = "bidir"
    none = "none"


def _record_iperf3(docker_controller: DoubleDockerController, iperf3_server_ip: str, tunnel: str, request_id: str,
                   test: Test):
    sh = (
        f"iperf3 "
        f"--time 30 "
        f"--length 1k "
        f"--set-mss 1460 "
        f"--bitrate 200m "
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
        sh += (
            f"--client {iperf3_server_ip} "
        )
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

    with NetworkCapture(docker_controller, interface_tun_name, request_id) as capture:
        safe_log(logger.info, "Starting test")
        log_stream(
            docker_controller.client.container.execute,
            "iperf3-client",
            ["sh", "-c", sh],
        )

    return capture.syn_ack_epoch


@app.task(bind=True)
def run(
        self: Task, connection, tunnel: Tunnel, net_em: NetEm, dns_resolvers: List[str], verbose: bool, test: Test,
        client_docker_host: str = None, server_docker_host: str = None
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    local = client_docker_host == server_docker_host

    if (not local) and (net_em is not NetEm.none):
        raise Exception("Network emulation is not supported in remote mode")

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

        docker_veth_names = get_docker_veth_names()
        for name in docker_veth_names.keys():
            if "dns-resolver" in name:
                apply_net_em(docker_veth_names[name], net_em=net_em)

    env = Env(dns_resolvers_ips, local=local, server_host=urlparse(server_docker_host).hostname if server_docker_host else server_docker_host)
    double_docker_controller = new_docker_controller(env, tunnel, client_host=client_docker_host, server_host=server_docker_host)

    safe_log(logger.info, "Bringing up dummy containers")
    log_stream_logs(double_docker_controller.client.compose.up, wait=True, services=["dummy-client"])
    log_stream_logs(double_docker_controller.server.compose.up, wait=True, services=["dummy-server"])

    if local and not local_dns_resolvers_docker_controller:
        # add network emulation to the client
        docker_veth_names = get_docker_veth_names()
        apply_net_em(docker_veth_names["dummy-client"], net_em=net_em)

    safe_log(logger.info, "Bringing up new containers")
    log_stream_logs(double_docker_controller.server.compose.up, wait=True)
    log_stream_logs(double_docker_controller.client.compose.up, wait=True)

    if test == Test.none:
        self.update_state(state=SUCCESS)

        return

    safe_log(logger.info, "Waiting for all components to initialize (10 seconds)")
    time.sleep(10)

    try:
        server_ip = _get_server_ip(double_docker_controller, server_docker_host)

        syn_ack_epoch = _record_iperf3(double_docker_controller, server_ip, tunnel, self.request.id, test)

        sync_db(connection, self.request.id, tunnel, dns_resolvers, test, net_em, syn_ack_epoch)

        self.update_state(state=SUCCESS)

        return
    except Exception as e:
        self.update_state(state=FAILURE, meta={"error": str(e)})

        log_stream(double_docker_controller.client.compose.logs)
        log_stream(double_docker_controller.server.compose.logs)
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
    def __init__(self, docker_controller: DoubleDockerController, interface_tun_name, file_name):
        self.docker_controller = docker_controller
        self.interface_tun_name = interface_tun_name
        self.file_name = file_name

    def __enter__(self):
        # Start the dumpcap in a background thread
        self.dumpcap_thread_client = threading.Thread(target=run_dumpcap, args=(self.docker_controller.client, "dumpcap-client", self.interface_tun_name, self.file_name))
        self.dumpcap_thread_client.start()

        self.dumpcap_thread_server = threading.Thread(target=run_dumpcap, args=(self.docker_controller.server, "dumpcap-server", self.interface_tun_name, self.file_name))
        self.dumpcap_thread_server.start()

        safe_log(logger.info, "Waiting for packet capture to initialize (10 seconds)")
        time.sleep(10)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        file_name = self.file_name


        safe_log(logger.info, "Waiting for wild packets to die (10 seconds)")
        time.sleep(10)

        safe_log(logger.info, "Stopping tshark process")

        log_stream(self.docker_controller.client.container.execute, "dumpcap-client", ["killall", "dumpcap"])
        log_stream(self.docker_controller.server.container.execute, "dumpcap-server", ["killall", "dumpcap"])

        # Wait for the thread to join, ensuring that `dumpcap` has finished
        self.dumpcap_thread_client.join(timeout=10)
        self.dumpcap_thread_server.join(timeout=10)

        log_stream(self.docker_controller.client.container.execute, "dumpcap-client", ["tcpdump", "-r", f"/{file_name}.pcap", "-c", "1", "tcp[tcpflags] & (tcp-syn|tcp-ack) == (tcp-syn|tcp-ack)", "-w", f"/{file_name}-handshake.pcap", "--time-stamp-precision", "nano"])
        self.syn_ack_epoch = self.docker_controller.client.container.execute("dumpcap-client", ["tshark", "-r", f"/{file_name}-handshake.pcap", "-T", "fields", "-e", "frame.time_epoch"])

        sh = (
            f"tshark -r /{file_name}.pcap -T fields "
            f"-e frame.time_epoch "
            f"-e frame.len "
            f"-e frame.interface_id "
            f"-e ip.src "
            f"-e ip.dst "
            f"-e tcp.srcport "
            f"-e tcp.dstport "
            f"-e udp.srcport "
            f"-e udp.dstport "
            f"-e ip.proto "
            f"-e tcp.flags "
            f"-E separator=, "
            f"-E quote=d "
            f"-E header=y > /{file_name}.csv"
        )

        safe_log(logger.info, "Running tshark to convert pcap to csv: sh -c '" + sh + "'")
        log_stream(self.docker_controller.client.container.execute, "dumpcap-client", ["sh", "-c", sh])
        log_stream(self.docker_controller.server.container.execute, "dumpcap-server", ["sh", "-c", sh])

        self.docker_controller.client.container.copy(source=("dumpcap-client", f"/{file_name}.pcap"), destination=f"./celery/artifacts/{file_name}-client.pcap")
        self.docker_controller.client.container.copy(source=("dumpcap-client", f"/{file_name}.csv"), destination=f"./celery/artifacts/{file_name}-client.csv")
        self.docker_controller.server.container.copy(source=("dumpcap-server", f"/{file_name}.pcap"), destination=f"./celery/artifacts/{file_name}-server.pcap")
        self.docker_controller.server.container.copy(source=("dumpcap-server", f"/{file_name}.csv"), destination=f"./celery/artifacts/{file_name}-server.csv")


def get_docker_veth_names() -> dict[str, str]:
    veth_names = {}
    dockerveth_cmd = "sudo sh tools/dockerveth/dockerveth.sh"
    safe_log(logger.info, f"Running dockerveth: {dockerveth_cmd}")
    output = subprocess.run(dockerveth_cmd.split(), check=True, capture_output=True).stdout.decode("utf-8")
    lines = output.splitlines()
    if lines[0].startswith("CONTAINER ID"):
        lines = lines[1:]
    for line in lines:
        container_id, veth, container_name = line.split()
        veth_names[container_name] = veth

    return veth_names


def apply_net_em(veth: str, net_em: NetEm):
    if net_em is None or net_em is net_em.none:
        return

    config = net_em.config
    delay = config["delay"]
    jitter = config["jitter"]
    loss = config["loss"]
    rate = config["rate"]

    netem_cmd = f"sudo tc qdisc add dev {veth} root netem delay {delay}ms {jitter} loss {loss}% rate {rate}mbit"
    safe_log(logger.info, f"Applying network emulation on {veth}: {netem_cmd}")
    subprocess.run(netem_cmd.split(), check=True)

    safe_log(logger.info, f"Network emulation applied successfully on {veth}")


def sync_db(connection, task_id: str, tunnel: Tunnel, dns_resolvers: List[str], test: Test, net_em: NetEm, syn_ack_epoch: float):
    with connection.cursor() as cursor:
        # Insert metadata into task_metadata table
        cursor.execute("""
                INSERT INTO task_metadata (task_id, tunnel_name, dns_resolvers, test_type, netem_name, netem)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (task_id, tunnel.value, dns_resolvers, test.value, net_em.value, json.dumps(net_em.config)))

        for side in ["client", "server"]:
            # Load packet data from CSV into packet_data table (without task_id initially)
            csv_file_path = f"./celery/artifacts/{task_id}-{side}.csv"
            with open(csv_file_path, 'r') as f:
                cursor.copy_expert("""
                    COPY packet_data (frame_time, frame_len, frame_interface_id, ip_src, ip_dst,
                                      tcp_srcport, tcp_dstport, udp_srcport, udp_dstport, ip_proto, tcp_flags)
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

        connection.commit()


