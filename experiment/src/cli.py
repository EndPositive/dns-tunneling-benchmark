from typing import Annotated

import typer

from .lib.lib import Tunnel, logger

cli = typer.Typer()
opt_tunnel: Tunnel = None
opt_verbose = False
opt_use_dns_resolver = False
opt_client_docker_host = None
opt_server_docker_host = None


@cli.command()
def destroy():
    from .tasks import destroy

    result = destroy.apply()
    print(f"Task result: {result}")

@cli.command()
def run(upload_test: bool = False, download_test: bool = False, bidir_test: bool = False):
    from .tasks import run

    if upload_test is False and download_test is False and bidir_test is False:
        raise Exception("No test selected")

    if download_test:
        logger.info("Running download test")
        result = run.apply(
            args=[opt_tunnel.value, opt_use_dns_resolver, opt_verbose, True, False, opt_client_docker_host, opt_server_docker_host],
        )
        logger.info(f"Task result: {result}")
    if upload_test:
        logger.info("Running upload test")
        result = run.apply(
            args=[opt_tunnel.value, opt_use_dns_resolver, opt_verbose, False, True, opt_client_docker_host, opt_server_docker_host],
        )
        logger.info(f"Task result: {result}")
    if bidir_test:
        logger.info("Running bidir test")
        result = run.apply(
            args=[opt_tunnel.value, opt_use_dns_resolver, opt_verbose, True, True, opt_client_docker_host, opt_server_docker_host],
        )
        logger.info(f"Task result: {result}")


@cli.command()
def restart():
    from .tasks import restart

    result = restart.apply(
        args=[opt_tunnel.value, opt_use_dns_resolver, opt_verbose, opt_client_docker_host, opt_server_docker_host],
    )
    logger.info(f"Task result: {result}")

@cli.command()
def build():
    from .tasks import build

    result = build.apply(
        args=[opt_tunnel.value, opt_use_dns_resolver, opt_verbose, opt_client_docker_host, opt_server_docker_host],
    )
    logger.info(f"Task result: {result}")


@cli.callback()
def main(
    tunnel: Annotated[Tunnel, typer.Option(case_sensitive=True)],
    verbose: bool = False,
    use_dns_resolver: bool = False,
    client_docker_host: str = None,
    server_docker_host: str = None,
):
    global opt_tunnel, opt_use_dns_resolver, opt_verbose, opt_client_docker_host, opt_server_docker_host
    opt_tunnel = tunnel
    opt_use_dns_resolver = use_dns_resolver
    opt_verbose = verbose
    opt_client_docker_host = client_docker_host
    opt_server_docker_host = server_docker_host
