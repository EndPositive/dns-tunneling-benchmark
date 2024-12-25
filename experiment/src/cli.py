from typing import Annotated, List
from urllib.parse import urlparse

import psycopg2
import typer
from python_on_whales import DockerClient

from .lib.docker import new_docker_controller, new_local_dns_resolvers_docker_controller
from .lib.lib import Env, Tunnel
from .lib.log import log_stream_logs, logger, safe_log
from .tasks.run import NetEm, Test

cli = typer.Typer()
opt_verbose = False
opt_client_docker_host = None
opt_server_docker_host = None

opt_type_test = Annotated[List[Test], typer.Option(case_sensitive=False, rich_help_panel="Experiment configuration")]
opt_type_tunnel = Annotated[List[Tunnel], typer.Option(case_sensitive=True, rich_help_panel="Experiment configuration")]
opt_type_dns_resolver = Annotated[List[str], typer.Option(help="DNS resolver", rich_help_panel="Experiment configuration")]
opt_type_net_em = Annotated[List[NetEm], typer.Option(
    case_sensitive=False,
    help="\n".join(f"{name}: {NetEm[name].config}" for name in NetEm.__members__),
    rich_help_panel="Network emulation",
)]
opt_type_db = Annotated[str, typer.Option(help="PostgreSQL connection string", rich_help_panel="Database")]
opt_type_docker_host = Annotated[str, typer.Option(help="Client-side Docker host (e.g. tcp://1.2.3.4:2376", rich_help_panel="Docker hosts")]

@cli.command()
def run(
        test: opt_type_test,
        tunnel: opt_type_tunnel,
        dns_resolver: opt_type_dns_resolver = [],
        net_em: opt_type_net_em = [NetEm.none],
        db: opt_type_db = "postgresql://username:password@localhost:5432/postgres",
):
    from .tasks.run import run

    if len(test) == 0:
        raise Exception("No test selected")

    connection = psycopg2.connect(db)

    for tunnel_t in tunnel:
        for net_em_t in net_em:
            for test_t in test:
                logger.info(f"Running {test_t.value} test")
                result = run.apply(
                    args=[connection, tunnel_t, net_em_t, dns_resolver, opt_verbose, test_t, opt_client_docker_host,
                          opt_server_docker_host],
                )
                logger.info(f"Task result: {result}")


@cli.command()
def destroy():
    safe_log(logger.info, f"Downing entire compose project")
    for host in {opt_client_docker_host, opt_server_docker_host}:
        downer = DockerClient(compose_files=["docker-compose-common.yaml"], host=host)
        log_stream_logs(downer.compose.down, timeout=1, remove_orphans=True)


@cli.command()
def build(
        tunnel: opt_type_tunnel,
        dns_resolver: opt_type_dns_resolver = [],
):
    local_dns_resolvers_docker_controller = new_local_dns_resolvers_docker_controller(dns_resolver)
    if local_dns_resolvers_docker_controller:
        local_dns_resolvers_docker_controller.compose.build(services="dns-resolver")

    env = Env([], local=opt_client_docker_host == opt_server_docker_host, server_host=urlparse(
        opt_server_docker_host).hostname if opt_server_docker_host else opt_server_docker_host)

    for tunnel_t in tunnel:
        docker_controller = new_docker_controller(env, tunnel_t, client_host=opt_client_docker_host,
                                                  server_host=opt_server_docker_host)

        log_stream_logs(docker_controller.client.compose.build, progress="plain")
        log_stream_logs(docker_controller.server.compose.build, progress="plain")


@cli.callback()
def main(
        client_docker_host: opt_type_docker_host = None,
        server_docker_host: opt_type_docker_host = None,
        verbose: bool = False,
):
    global opt_client_docker_host, opt_server_docker_host, opt_verbose
    opt_client_docker_host = client_docker_host
    opt_server_docker_host = server_docker_host
    opt_verbose = verbose
