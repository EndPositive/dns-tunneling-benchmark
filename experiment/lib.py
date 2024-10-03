import logging
from getpass import getpass
import subprocess
import tempfile
from contextlib import contextmanager
from enum import Enum

from python_on_whales import Network

SERVICES = [
    "big-files",
    "socks-server",
    "socks-client",
    "dns-resolver",
    "dns-tunnel-server",
    "dns-tunnel-client",
]
env = {
    "IPV4_ADDRESS_IPERF3_SERVER": "172.22.0.1",
    "IPV4_ADDRESS_SOCKS_CLIENT": "172.22.0.2",
    "IPV4_ADDRESS_DNS_TUNNEL_CLIENT": "172.22.0.3",
    "IPV4_ADDRESS_DNS_RESOLVER": "172.22.0.4",
    "IPV4_ADDRESS_DNS_TUNNEL_SERVER": "172.22.0.5",
    "IPV4_ADDRESS_SOCKS_SERVER": "172.22.0.6",
    "IPV4_ADDRESS_BIG_FILES": "172.22.0.7",
}
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class Tunnel(str, Enum):
    dns2tcp = "dns2tcp"
    dnscat2 = "dnscat2"
    dnstt = "dnstt"
    iodine = "iodine"
    ozyman = "OzymanDNS"
    tuns = "TUNS"


class FileSize(str, Enum):
    one_byte = "1b"
    one_kilobyte = "1kb"
    one_megabyte = "1mb"
    ten_megabytes = "10mb"
    one_hundred_megabytes = "100mb"


@contextmanager
def start_tshark_process(interface, filename):
    logger.info("Starting tshark process")
    tshark_process = subprocess.Popen(['tshark', '-i', interface, '-w', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield tshark_process
    finally:
        logger.info("Stopping tshark process")
        tshark_process.terminate()
        tshark_process.wait()


def stream_logs(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream_logs=True)
    for _, stream_content in output_stream:
        logger.info(stream_content.decode("utf-8").strip())


def stream(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream=True)
    for _, stream_content in output_stream:
        logger.info(stream_content.decode("utf-8").strip())


def get_interface_name(network: Network):
    interface_name = f"br-{network.id[:12]}"
    return interface_name

def get_veth_names():
    password = getpass("Please enter your password: ")
    result = subprocess.run(["sudo", "-S", "sh", "./tools/dockerveth/dockerveth.sh"], input=password + "\n", capture_output=True, text=True, check=True)
    lines = result.stdout.split("\n")
    veth_names = {}
    for line in lines[1:]:
        if len(line) == 0:
            continue
        container_id, veth, name = line.split()
        veth_names[name] = veth
    return veth_names


def apply_netem(veth: str, delay_ms: int = 183, jitter_ms: int = 15, loss_percentage: int = 9):
    subprocess.run(f"sudo tc qdisc add dev {veth} root handle 1: tbf rate 21mbit burst 32kbit latency 50ms".split(), check=True)
    subprocess.run(f"sudo tc qdisc add dev {veth} root netem delay {delay_ms}ms {jitter_ms}ms loss {loss_percentage}%".split(), check=True)

def reset_netem(veth: str):
    subprocess.run(f"tc qdisc del dev {veth} root".split(), check=True)

def create_tmp_env_file(use_dns_resolver: bool) -> str:
    if not use_dns_resolver:
        env["IPV4_ADDRESS_DNS_RESOLVER"] = env["IPV4_ADDRESS_DNS_TUNNEL_SERVER"]
        SERVICES.remove("dns-resolver")

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".env"
    ) as temp_env:
        for key, value in env.items():
            temp_env.write(f"{key}={value}\n")
        temp_env_filename = temp_env.name

    return temp_env_filename
