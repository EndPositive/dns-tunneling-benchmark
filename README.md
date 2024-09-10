# DNS Tunneling Benchmark Docker

## Experiment

The experiments can be ran using `experiment.py`.
First install the required python dependencies from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Then you can run the experiment using the following command.

```bash
python experiment.py --help
```

### Network Simulation

The created docker network can be emulated using the Linux netem module.
First, you need to know the virtual interface name of the container.
We can use the following script to retrieve this.

```bash
sudo sh tools/dockerveth/dockerveth.sh
CONTAINER ID    VETH            NAMES
65a34b856518    veth349fc67     tuns-dns-tunnel-client-1
12913dbbdce6    veth7fe455a     tuns-socks-client-1
ef159eb474f1    vetha262d9a     tuns-dns-tunnel-server-1
0d9345b1fef7    veth8caa5d4     tuns-socks-server-1
2b73f6a4feed    vethc56f044     tuns-iperf3-server-1
```

Then it is possible to add network emulation to the virtual interface.
For instance, to add packet loss to the DNS tunnel client, we can use the following command.

```bash
sudo tc qdisc add dev veth349fc67 root netem loss 10%
```

To remove the network emulation, we can use the following command.

```bash
sudo tc qdisc del dev veth349fc67 root
```
