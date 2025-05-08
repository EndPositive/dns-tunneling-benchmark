from typing import List

from python_on_whales import DockerClient

from ..lib.lib import Env, Tunnel, write_dict_to_env_file
from ..lib.log import logger

def new_local_dns_resolvers_docker_controller(dns_resolvers: List[str]) -> DockerClient | None:
    local_dns_resolvers_nb = dns_resolvers.count("local")
    if local_dns_resolvers_nb == 0:
        return None

    local_dns_resolvers_docker_controller = DockerClient(
        compose_files=["docker-compose-common.yaml", "docker-compose-resolver.yaml"],
        compose_env_files=[write_dict_to_env_file({"DNS_RESOLVER_REPLICAS": local_dns_resolvers_nb})],
    )
    return local_dns_resolvers_docker_controller


class DoubleDockerController:
    def __init__(self, client: DockerClient, server: DockerClient, local=True):
        self.client = client
        self.server = server
        self.local = local


def new_docker_controller(env: Env, tunnel: Tunnel, client_host: str = None,
                          server_host: str = None) -> DoubleDockerController:
    logger.info(f"Creating a double docker client ({client_host}, {server_host})")

    def _new_docker_controller(client: bool, server: bool, host=None) -> DockerClient:
        tmp_env_filename = env.create_tmp_env_file()

        compose_files = [
            "docker-compose-common.yaml"
        ]

        if client:
            compose_files.append("docker-compose-client-browsing.yaml")
            compose_files.append("docker-compose-client-dummy.yaml")
            compose_files.append("docker-compose-client-dumpcap.yaml")
            compose_files.append("docker-compose-client-file-transfer.yaml")
            compose_files.append("docker-compose-client-iperf3.yaml")
            compose_files.append("docker-compose-client-speedtest-cli.yaml")
            compose_files.append(f"tunnels/{tunnel.value}/docker-compose-client.yaml")

        if server:
            compose_files.append("docker-compose-server-dummy.yaml")
            compose_files.append("docker-compose-server-dumpcap.yaml")
            compose_files.append("docker-compose-server-file-transfer.yaml")
            compose_files.append("docker-compose-server-iperf3.yaml")
            compose_files.append("docker-compose-server-iperf3-dumpcap.yaml")
            compose_files.append("docker-compose-server-speedtest-server.yaml")
            compose_files.append("docker-compose-server-websites.yaml")
            compose_files.append(f"tunnels/{tunnel.value}/docker-compose-server.yaml")

        return DockerClient(
            host=host if host != "local" else None,
            compose_files=compose_files,
            compose_env_files=[tmp_env_filename],
        )

    local = client_host == server_host
    return DoubleDockerController(
        _new_docker_controller(client=True, server=False, host=client_host),
        _new_docker_controller(client=False, server=True, host=server_host),
        local=local,
    )
