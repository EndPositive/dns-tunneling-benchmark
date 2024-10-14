from typing import Annotated

import typer

from .lib.lib import Tunnel

cli = typer.Typer()
opt_tunnel: Tunnel = None
opt_verbose = False
opt_use_dns_resolver = False


@cli.command()
def destroy():
    from .tasks import destroy

    result = destroy.apply()
    print(f"Task result: {result}")


@cli.command()
def run_async():
    task = run.apply_async(
        args=[opt_tunnel.value, opt_use_dns_resolver, opt_verbose],
    )
    result = task.get(timeout=60)
    print(f"Task result: {result}")


@cli.command()
def run():
    from .tasks import run

    result = run.apply(
        args=[opt_tunnel.value, opt_use_dns_resolver, opt_verbose],
    )
    print(f"Task result: {result}")


@cli.callback()
def main(
    tunnel: Annotated[Tunnel, typer.Option(case_sensitive=True)],
    verbose: bool = False,
    use_dns_resolver: bool = False,
):
    global opt_tunnel, opt_use_dns_resolver, opt_verbose
    opt_tunnel = tunnel
    opt_use_dns_resolver = use_dns_resolver
    opt_verbose = verbose
