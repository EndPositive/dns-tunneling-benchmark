import tempfile
from enum import Enum
from typing import List


class Env:
    IPV4_ADDRESS_DNS_TUNNEL_CLIENT = "172.22.0.4"
    IPV4_ADDRESS_DNS_TUNNEL_SERVER = "172.22.0.6"
    IPV4_ADDRESS_IPERF3_SERVER = "172.22.0.7"

    IPV4_ADDRESS_DNS_RESOLVER: str = None

    def __init__(self, dns_resolvers_ips: List[str], local, server_host):
        if len(dns_resolvers_ips) == 1:
            self.IPV4_ADDRESS_DNS_RESOLVER = dns_resolvers_ips[0]
        elif len(dns_resolvers_ips) == 0:
           if not local:
                self.IPV4_ADDRESS_DNS_RESOLVER = server_host
           else:
                self.IPV4_ADDRESS_DNS_RESOLVER = self.IPV4_ADDRESS_DNS_TUNNEL_SERVER
        else:
            # TODO: support more than 1 DNS resolver
            pass

    def create_tmp_env_file(self) -> str:
        env = {key: value for key, value in self.__class__.__dict__.items() if
               not key.startswith('__') and not callable(value)}
        env.update(
            {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)})

        return write_dict_to_env_file(env)

def write_dict_to_env_file(env: dict) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".env") as temp_env:
        for key, value in env.items():
            temp_env.write(f"{key}={value}\n")
        temp_env_filename = temp_env.name

    return temp_env_filename


class Tunnel(str, Enum):
    dnstunnler = "fd/dns-tunnler"
    dnscapy = "fd/dnscapy"
    dnstunnel = "fd/dnstunnel"
    ozyman = "fd/OzymanDNS"
    sods = "fd/sods"
    dns2tcp = "tcp/dns2tcp"
    dnscat2 = "tcp/dnscat2"
    dnstt = "tcp/dnstt"
    dnstt_quic_cw = "tcp/dnstt-quic-cw"
    slipstream = "tcp/slipstream"
    iodine = "tun/iodine"
    tuns = "tun/TUNS"
    raw = "raw"


