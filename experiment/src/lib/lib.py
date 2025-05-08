import tempfile
from enum import Enum
from typing import List

from src.lib.tests import Test


class Env:
    IPV4_ADDRESS_DNS_TUNNEL_CLIENT = "172.22.0.15"
    IPV4_ADDRESS_DNS_TUNNEL_SERVER = "172.22.0.16"
    IPV4_ADDRESS_IPERF3_SERVER = "172.22.0.17"
    IPV4_ADDRESS_SPEEDTEST_SERVER = "172.22.0.18"
    IPV4_ADDRESS_NC_SERVER = "172.22.0.19"
    IPV4_ADDRESS_WEBSITES_SERVER = "172.22.0.20"

    IPV4_ADDRESS_DNS_TUNNEL_TARGET: str = None
    IPV4_ADDRESS_DNS_RESOLVER: str = None
    IPV4_ADDRESS_DNS_RESOLVERS_SLIPSTREAM: str = None

    def __init__(self, dns_resolvers_ips: List[str], local, server_host, test):
        if len(dns_resolvers_ips) == 1:
            self.IPV4_ADDRESS_DNS_RESOLVER = dns_resolvers_ips[0]
            self.IPV4_ADDRESS_DNS_RESOLVERS_SLIPSTREAM = f"{self.IPV4_ADDRESS_DNS_RESOLVER} 53"
        elif len(dns_resolvers_ips) == 0:
            if not local:
                self.IPV4_ADDRESS_DNS_RESOLVER = server_host
            else:
                self.IPV4_ADDRESS_DNS_RESOLVER = self.IPV4_ADDRESS_DNS_TUNNEL_SERVER
            self.IPV4_ADDRESS_DNS_RESOLVERS_SLIPSTREAM = f"{self.IPV4_ADDRESS_DNS_RESOLVER} 53"
        else:
            self.IPV4_ADDRESS_DNS_RESOLVERS_SLIPSTREAM = "\n".join([f"{ip} 53" for ip in dns_resolvers_ips])

        if test == Test.latency:
            self.IPV4_ADDRESS_DNS_TUNNEL_TARGET = self.IPV4_ADDRESS_SPEEDTEST_SERVER
        elif test == Test.file_download:
            self.IPV4_ADDRESS_DNS_TUNNEL_TARGET = self.IPV4_ADDRESS_NC_SERVER
        elif test == Test.file_upload:
            self.IPV4_ADDRESS_DNS_TUNNEL_TARGET = self.IPV4_ADDRESS_NC_SERVER
        elif test == Test.browsing:
            self.IPV4_ADDRESS_DNS_TUNNEL_TARGET = self.IPV4_ADDRESS_WEBSITES_SERVER
        else:
            self.IPV4_ADDRESS_DNS_TUNNEL_TARGET = self.IPV4_ADDRESS_IPERF3_SERVER


    def create_tmp_env_file(self) -> str:
        env = {key: value for key, value in self.__class__.__dict__.items() if
               not key.startswith('__') and not callable(value)}
        env.update(
            {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)})

        return write_dict_to_env_file(env)

def write_dict_to_env_file(env: dict) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".env") as temp_env:
        for key, value in env.items():
            temp_env.write(f"{key}='{value}'\n")
        temp_env_filename = temp_env.name

    return temp_env_filename


class Tunnel(str, Enum):
    slipstream = "tcp/slipstream"
    dnstt_quic = "tcp/dnstt-quic"
    dnstt = "tcp/dnstt"
    dnstunnler = "fd/dns-tunnler"
    dnscapy = "fd/dnscapy"
    dnstunnel = "fd/dnstunnel"
    ozyman = "fd/OzymanDNS"
    sods = "fd/sods"
    dns2tcp = "tcp/dns2tcp"
    dnscat2 = "tcp/dnscat2"
    iodine = "tun/iodine"
    tuns = "tun/TUNS"
    raw = "raw"


