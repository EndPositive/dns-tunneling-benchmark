import tempfile
import time
from contextlib import contextmanager

import typer
from python_on_whales import DockerClient, Network
from scapy.all import conf
from scapy.sendrecv import AsyncSniffer
from scapy.utils import wrpcap

app = typer.Typer()
opt_verbose = False
opt_destroy = False
opt_use_dns_resolver = False

conf.use_pcap = True

env = {
    "IPV4_ADDRESS_SOCKS_SERVER": "172.22.0.6",
    "IPV4_ADDRESS_BIG_FILES": "172.22.0.7",
    "IPV4_ADDRESS_SOCKS_CLIENT": "172.22.0.5",
    "IPV4_ADDRESS_IPERF3_SERVER": "172.22.0.4",
    "IPV4_ADDRESS_DNS_TUNNEL_CLIENT": "172.22.0.3",
    "IPV4_ADDRESS_DNS_TUNNEL_SERVER": "172.22.0.2",
    "IPV4_ADDRESS_DNS_RESOLVER": "172.22.0.9",
}

SERVICES = [
    "big-files",
    "socks-server",
    "socks-client",
    "dns-resolver",
    "dns-tunnel-server",
    "dns-tunnel-client",
]


@contextmanager
def packet_sniffer(interface, filename):
    """Context manager to handle packet sniffing."""
    sniffer = AsyncSniffer(iface=interface, count=0)
    print("Starting packet sniffer")
    sniffer.start()
    time.sleep(1)
    try:
        yield sniffer
    finally:
        print("Stopping packet sniffer")
        time.sleep(1)
        packets = sniffer.stop()
        wrpcap(filename, packets)
        print(f"Captured packets saved to {filename}")


def stream_logs(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream_logs=True)

    for _, stream_content in output_stream:
        print(stream_content.decode("utf-8"), end="")


def stream(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream=True)

    for _, stream_content in output_stream:
        print(stream_content.decode("utf-8"), end="")


def get_interface_name(network: Network):
    interface_name = f"br-{network.id[:12]}"
    return interface_name


def run_experiment(tunnel_name: str):
    if not opt_use_dns_resolver:
        env["IPV4_ADDRESS_DNS_RESOLVER"] = env["IPV4_ADDRESS_DNS_TUNNEL_SERVER"]
        SERVICES.remove("dns-resolver")

    # Create a temporary .env file
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".env"
    ) as temp_env:
        for key, value in env.items():
            temp_env.write(f"{key}={value}\n")

        # Save the file name for later use
        temp_env_filename = temp_env.name

    docker_client = DockerClient(
        compose_files=[
            "docker-compose.yaml",
            f"tunnels/{tunnel_name}/docker-compose.yaml",
        ],
        compose_env_files=[temp_env_filename],
    )

    if opt_destroy:
        docker_client.compose.down(volumes=True, timeout=1)
        return

    # docker_client.compose.build()

    print("Downing any running containers")
    stream_logs(docker_client.compose.down, timeout=1)

    print("Bringing up new containers")
    stream_logs(docker_client.compose.up, services=SERVICES, wait=True, color=True)

    print("Inspecting created network")
    network = docker_client.network.inspect("benchmark_experiment")

    # Countdown timer before sniffing
    print("Waiting for all components to initialize (10 seconds)")
    time.sleep(10)

    file_name = f"pcap-{"dns-resolver" if opt_use_dns_resolver else "no-dns-resolver"}-{tunnel_name}-{time.strftime('%Y%m%d-%H%M%S')}.pcap"

    with packet_sniffer(
        get_interface_name(network),
        file_name,
    ):
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
                "1mb.txt",
            ],
        )


@app.command()
def dns2tcp():
    return run_experiment("dns2tcp")


@app.command()
def dnscat2():
    return run_experiment("dnscat2")


@app.command()
def dnstt():
    return run_experiment("dnstt")


@app.command()
def iodine():
    return run_experiment("iodine")


@app.command()
def ozyman():
    return run_experiment("OzymanDNS")


@app.command()
def tuns():
    return run_experiment("TUNS")


@app.callback()
def main(verbose: bool = False, destroy: bool = False, use_dns_resolver: bool = False):
    global opt_verbose
    global opt_destroy
    global opt_use_dns_resolver
    if verbose:
        opt_verbose = True
    if destroy:
        opt_destroy = True
    if use_dns_resolver:
        opt_use_dns_resolver = True


if __name__ == "__main__":
    app()
