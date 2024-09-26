#!/usr/bin/env python3
from typing import Annotated

import typer
from python_on_whales import DockerClient

from lib import FileSize, Tunnel, create_tmp_env_file, stream_logs
from tasks import run_experiment

cli = typer.Typer()
opt_tunnel: Tunnel = None
opt_verbose = False
opt_use_dns_resolver = False
opt_file_size = FileSize.one_megabyte


@cli.command()
def destroy():
    tmp_env_file = create_tmp_env_file(opt_use_dns_resolver)

    docker_client = DockerClient(
        compose_files=[
            "docker-compose.yaml",
            f"tunnels/{opt_tunnel.name}/docker-compose.yaml",
        ],
        compose_env_files=[tmp_env_file],
    )

    stream_logs(docker_client.compose.down, volumes=True, timeout=1)


@cli.command()
def run():
    task = run_experiment.apply_async(
        args=[opt_tunnel.value, opt_use_dns_resolver, opt_file_size.value, opt_verbose],
    )
    result = task.get(timeout=60)
    print(f"Task result: {result}")



@cli.callback()
def main(
    tunnel: Annotated[Tunnel, typer.Option(case_sensitive=True)],
    verbose: bool = False,
    use_dns_resolver: bool = False,
    file_size: Annotated[
        FileSize, typer.Option(case_sensitive=True)
    ] = FileSize.one_megabyte,
):
    global opt_tunnel, opt_use_dns_resolver, opt_file_size, opt_verbose
    opt_tunnel = tunnel
    opt_use_dns_resolver = use_dns_resolver
    opt_file_size = file_size
    opt_verbose = verbose


if __name__ == "__main__":
    cli()
