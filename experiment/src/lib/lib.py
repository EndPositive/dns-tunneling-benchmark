import logging
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from enum import Enum

from python_on_whales import DockerClient, Network

# Create a global lock
log_lock = threading.Lock()

class Env:
    IPV4_ADDRESS_GATEWAY = "172.22.0.1"
    IPV4_ADDRESS_DNS_TUNNEL_CLIENT = "172.22.0.4"
    IPV4_ADDRESS_DNS_RESOLVER = "172.22.0.5"
    IPV4_ADDRESS_DNS_TUNNEL_SERVER = "172.22.0.6"
    IPV4_ADDRESS_IPERF3_SERVER = "172.22.0.8"

    IPV4_ADDRESS_DNS_TUNNEL_CLIENT_TARGET_RESOLVER: str = None
    IPV4_ADDRESS_SOCKS_SERVER_TARGET: str = None

    def __init__(self, use_dns_resolver: bool, raw: bool, local, server_host):
        if use_dns_resolver:
            self.IPV4_ADDRESS_DNS_TUNNEL_CLIENT_TARGET_RESOLVER = self.IPV4_ADDRESS_DNS_RESOLVER
        elif not local:
            self.IPV4_ADDRESS_DNS_TUNNEL_CLIENT_TARGET_RESOLVER = server_host
        else:
            self.IPV4_ADDRESS_DNS_TUNNEL_CLIENT_TARGET_RESOLVER = self.IPV4_ADDRESS_DNS_TUNNEL_SERVER

        if raw:
            self.IPV4_ADDRESS_SOCKS_SERVER_TARGET = self.IPV4_ADDRESS_DNS_TUNNEL_SERVER
        else:
            self.IPV4_ADDRESS_SOCKS_SERVER_TARGET = "127.0.0.1"

    def create_tmp_env_file(self) -> str:
        env = {key: value for key, value in self.__class__.__dict__.items() if
               not key.startswith('__') and not callable(value)}
        env.update(
            {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)})

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".env") as temp_env:
            for key, value in env.items():
                temp_env.write(f"{key}={value}\n")
            temp_env_filename = temp_env.name

        return temp_env_filename


logging.basicConfig(
    format='%(asctime)s - %(name)s:%(lineno)s:%(funcName)20s() - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class Tunnel(str, Enum):
    dnstunnler = "socks/dns-tunnler"
    dns2tcp = "socks/dns2tcp"
    dnscapy = "socks/dnscapy"
    dnstt = "socks/dnstt"
    dnstt_quic = "socks/dnstt-quic"
    iodine = "tun/iodine"
    ozyman = "socks/OzymanDNS"
    sods = "socks/sods"
    tuns = "tun/TUNS"
    raw = "raw"

class DoubleDockerController:
    def __init__(self, client: DockerClient, server: DockerClient, local=True):
        self.client = client
        self.server = server
        self.local = local


def new_docker_controller(env: Env, tunnel: str, use_dns_resolver: bool, client_host: str = None,
                          server_host: str = None) -> DoubleDockerController:
    logger.info(f"Creating a double docker client ({client_host}, {server_host})")
    def _new_docker_controller(client: bool, server: bool, host=None) -> DockerClient:
        tmp_env_filename = env.create_tmp_env_file()

        compose_files = [
            "docker-compose-common.yaml"
        ]

        if client:
            compose_files.append("docker-compose-client-dummy.yaml")
            compose_files.append("docker-compose-client-iperf3.yaml")
            compose_files.append("docker-compose-client-dumpcap.yaml")
            compose_files.append(f"tunnels/{tunnel}/docker-compose-client.yaml")

        if server:
            compose_files.append("docker-compose-server-iperf3.yaml")
            compose_files.append(f"tunnels/{tunnel}/docker-compose-server.yaml")

        # TODO: use dns resolver
        if use_dns_resolver:
            compose_files.append("docker-compose-resolver.yaml")

        return DockerClient(
            host=host if host != "local" else None,
            compose_files=compose_files,
            compose_env_files=[tmp_env_filename],
        )

    return DoubleDockerController(
        _new_docker_controller(client=True, server=False, host=client_host),
        _new_docker_controller(client=False, server=True, host=server_host),
        local=client_host == server_host,
    )



def container_pid(controller: DockerClient, container_name: str) -> int:
    return controller.container.inspect(container_name).state.pid

@contextmanager
def network_capture(docker_controller: DoubleDockerController, interface_tun_name, file_name):
    # Function to run `dumpcap` in a separate thread
    def run_dumpcap():
        safe_log(logger.info, "Running dumpcap in background")

        sh = [
            "dumpcap",
            "-w", f"{file_name}.pcap",
            "-B", "8192",
            "-q",
            "-i", "eth0",
            "-i", "lo",
        ]

        if interface_tun_name:
            sh.extend(["-i", interface_tun_name])

        try:
            log_stream(docker_controller.client.container.execute,
                       "dumpcap-client", sh)
        except Exception as e:
            logger.error(f"dumpcap process failed: {e}")

    # Start the dumpcap in a background thread
    dumpcap_thread = threading.Thread(target=run_dumpcap)
    dumpcap_thread.start()

    safe_log(logger.info, "Waiting for packet capture to initialize (10 seconds)")
    time.sleep(10)

    try:
        yield  # Allow the code inside the context manager to run
    finally:
        safe_log(logger.info, "Waiting for wild packets to die (10 seconds)")
        time.sleep(10)

        safe_log(logger.info, "Stopping tshark process")

        log_stream(docker_controller.client.container.execute, "dumpcap-client", ["killall", "dumpcap"])

        # Wait for the thread to join, ensuring that `dumpcap` has finished
        dumpcap_thread.join(timeout=10)

        sh = (
            f"tshark -r {file_name}.pcap -T fields "
            f"-e frame.time_relative "
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
            f"-E header=y > {file_name}.csv"
        )

        safe_log(logger.info, "Running tshark to convert pcap to csv: sh -c '" + sh + "'")
        log_stream(docker_controller.client.container.execute, "dumpcap-client", ["sh", "-c", sh])


def safe_log(fn, *args, **kwargs):
    with log_lock:
        kwargs['stacklevel'] = kwargs.get('stacklevel', 2)
        return fn(*args, **kwargs)

def log_stream_logs(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream_logs=True)
    for _, stream_content in output_stream:
        message = stream_content.decode("utf-8").strip()
        for line in message.splitlines():
            safe_log(logger.info, line, stacklevel=3)


def log_stream(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream=True)
    for _, stream_content in output_stream:
        message = stream_content.decode("utf-8").strip()
        for line in message.splitlines():
            safe_log(logger.info, line, stacklevel=3)


def return_stream(fn, *args, **kwargs) -> str:
    output_stream = fn(*args, **kwargs, stream=True)
    output = ""
    for _, stream_content in output_stream:
        output += stream_content.decode("utf-8")
    return output


def get_interface_name(network: Network):
    interface_name = f"br-{network.id[:12]}"
    return interface_name


def apply_netem(veth: str, tbf_bandwidth_mbit: int, tbf_burst: int, tbf_latency: int,
                netem_delay_ms: int, netem_jitter_ms: int, netem_loss_percentage: int,
                src_ip: str, dst_ip: str):
    tbf_cmd = f"sudo tc qdisc add dev {veth} root handle 1: tbf rate {tbf_bandwidth_mbit}mbit burst {tbf_burst}kb latency {tbf_latency}ms"
    safe_log(logger.info, f"Applying TBF on {veth}: {tbf_cmd}")
    subprocess.run(tbf_cmd.split(), check=True)

    netem_cmd = f"sudo tc qdisc add dev {veth} parent 1: handle 10: netem delay {netem_delay_ms}ms {netem_jitter_ms}ms loss gemodel {netem_loss_percentage}%"
    safe_log(logger.info, f"Applying network emulation on {veth}: {netem_cmd}")
    subprocess.run(netem_cmd.split(), check=True)

    filter_cmd = f"sudo tc filter add dev {veth} protocol ip parent 1:0 prio 1 u32 match ip src {src_ip} match ip dst {dst_ip} flowid 1:10"
    safe_log(logger.info, f"Applying IP filter on {veth}: {filter_cmd}")
    subprocess.run(filter_cmd.split(), check=True)

    safe_log(logger.info, f"Network emulation applied successfully on {veth}")
