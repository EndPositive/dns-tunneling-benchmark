import sys
import time
from typing import Iterable, Tuple

from python_on_whales import DockerClient, Network
from scapy.all import conf
from scapy.sendrecv import AsyncSniffer
from scapy.utils import wrpcap
from tqdm import tqdm

conf.use_pcap = True

def get_interface_name(network: Network):
    interface_name = f"br-{network.id[:12]}"
    return interface_name

def countdown_timer(seconds):
    for _ in tqdm(range(seconds, 0, -1), desc="Countdown", unit="s"):
        time.sleep(1)

def main():
    docker_client_network = DockerClient(
        compose_files=[
            "../../experiment/tasks/docker-compose.yaml",
            "./docker-compose.yaml"
        ])

    if len(sys.argv) > 1 and sys.argv[1] == "down":
        try:
            docker_client_network.compose.down(volumes=True)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            docker_client_network.compose.down(timeout=1)
        return

    docker_client_network.compose.build()

    try:
        docker_client_network.compose.up(services=["cadvisor"], wait=True)
        docker_client_network.compose.up(wait=True)

        # Countdown timer before sniffing
        countdown_timer(10)

        network = docker_client_network.network.inspect("tasks_experiment")

        # Perform packet sniffing
        a = AsyncSniffer(iface=get_interface_name(network), count=0)
        a.start()

        # Run docker exec asynchronously
        output_stream: Iterable[Tuple[str, bytes]] = docker_client_network.container.execute(
            "socks-client",
            [
                "curl", "--interface",
                "tun0",
                "http://172.22.0.7:8000/1mb.txt",
                "-vvv", "--output",
                "1mb.txt"
            ],
            stream=True
        )

        for _, stream_content in output_stream:
            print(stream_content.decode("utf-8"), end="")

        packets = a.stop()

        # Save captured packets
        wrpcap("packets.pcap", packets)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
