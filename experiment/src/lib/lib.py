import logging
import subprocess
import tempfile
from contextlib import contextmanager
from enum import Enum

from python_on_whales import Container, DockerClient, Network


class Env:
    IPV4_ADDRESS_GATEWAY = "172.22.0.1"
    IPV4_ADDRESS_DNS_TUNNEL_CLIENT = "172.22.0.4"
    IPV4_ADDRESS_DNS_RESOLVER = "172.22.0.5"
    IPV4_ADDRESS_DNS_TUNNEL_SERVER = "172.22.0.6"
    IPV4_ADDRESS_IPERF3_SERVER = "172.22.0.8"

    IPV4_ADDRESS_DNS_TUNNEL_CLIENT_TARGET_RESOLVER: str = None
    IPV4_ADDRESS_SOCKS_SERVER_TARGET: str = None

    def __init__(self, use_dns_resolver: bool = False, raw: bool = False):
        if use_dns_resolver:
            self.IPV4_ADDRESS_DNS_TUNNEL_CLIENT_TARGET_RESOLVER = self.IPV4_ADDRESS_DNS_RESOLVER
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


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Tunnel(str, Enum):
    dns2tcp = "socks/dns2tcp"
    dnstt = "socks/dnstt"
    iodine = "tun/iodine"
    ozyman = "socks/OzymanDNS"
    tuns = "socks/TUNS"
    raw = "raw"

def container_pid(controller: DockerClient, container_name: str) -> int:
    return controller.container.inspect(container_name).state.pid

@contextmanager
def start_tshark_process(interface_name_tun, interface_eth_name, pid, filename):
    logger.info("Starting tshark process")

    sh = (
        f"sudo nsenter -t {pid} -n "
        f"tshark -i {interface_name_tun} -i {interface_eth_name} -T fields "
        f"-e frame.number "
        f"-e frame.time_relative "
        f"-e frame.len "
        f"-e frame.interface_id "
        f"-e ip.src "
        f"-e ip.dst "
        f"-e ip.proto "
        f"-e ip.len "
        f"-e tcp.flags "
        f"-e eth.type "
        f"-e eth.len "
        f"-E separator=, "
        f"-E quote=d "
        f"-E header=y > {filename}"
    )

    logger.info("sh -c" + sh)

    tshark_process = subprocess.Popen(["sh", "-c", sh], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield tshark_process
    finally:
        logger.info("Stopping tshark process")
        tshark_process.terminate()
        tshark_process.wait()

def log_stream_logs(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream_logs=True)
    for _, stream_content in output_stream:
        logger.info(stream_content.decode("utf-8").strip())


def log_stream(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream=True)
    for _, stream_content in output_stream:
        logger.info(stream_content.decode("utf-8").strip())


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
    logger.info(f"Applying TBF on {veth}: {tbf_cmd}")
    subprocess.run(tbf_cmd.split(), check=True)

    netem_cmd = f"sudo tc qdisc add dev {veth} parent 1: handle 10: netem delay {netem_delay_ms}ms {netem_jitter_ms}ms loss gemodel {netem_loss_percentage}%"
    logger.info(f"Applying network emulation on {veth}: {netem_cmd}")
    subprocess.run(netem_cmd.split(), check=True)

    filter_cmd = f"sudo tc filter add dev {veth} protocol ip parent 1:0 prio 1 u32 match ip src {src_ip} match ip dst {dst_ip} flowid 1:10"
    logger.info(f"Applying IP filter on {veth}: {filter_cmd}")
    subprocess.run(filter_cmd.split(), check=True)

    logger.info(f"Network emulation applied successfully on {veth}")


class DoubleDockerClient:
    def __init__(self, client: DockerClient, server: DockerClient):
        self.client = client
        self.server = server

    def __getattr__(self, name):
        def method(*args, **kwargs):
            client_result = getattr(self.client, name)(*args, **kwargs)
            if self.client != self.server:
                server_result = getattr(self.server, name)(*args, **kwargs)
                return client_result, server_result
            return client_result
        return method

def new_docker_controller(env: Env, tunnel: str, use_dns_resolver: bool, client_host: str = None, server_host: str = None) -> DockerClient:
    def _new_docker_controller(client: bool, server: bool, host = None) -> DockerClient:
        tmp_env_filename = env.create_tmp_env_file()

        compose_files = [
            "docker-compose-common.yaml"
        ]

        if client:
            if tunnel.startswith("socks/") or tunnel == Tunnel.raw:
                compose_files.append("docker-compose-client-tun2socks.yaml")

            compose_files.append("docker-compose-client-iperf3.yaml")

        if server:
            if tunnel.startswith("socks/") or tunnel == Tunnel.raw:
                compose_files.append("docker-compose-server-socks.yaml")

            compose_files.append("docker-compose-server-iperf3.yaml")

        if client:
            compose_files.append(f"tunnels/{tunnel}/docker-compose-client.yaml")
        if server:
            compose_files.append(f"tunnels/{tunnel}/docker-compose-server.yaml")

        # TODO: use dns resolver
        if use_dns_resolver:
            compose_files.append("docker-compose-resolver.yaml")

        return DockerClient(
            host=host if host != "local" else None,
            compose_files=compose_files,
            compose_env_files=[tmp_env_filename],
        )


    if client_host != server_host:
        return DoubleDockerClient(
            _new_docker_controller(client=True, server=False, host=client_host),
            _new_docker_controller(client=False, server=True, host=server_host)
        )
    else:
        return _new_docker_controller(client=True, server=True, host=client_host)












