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
sudo sh tools/docker-veth/docker-veth.sh 
veth28ff3c8@if2 3a139628b3cc dummy-server
vethf4ef9dc@if2 71cf6d6b201c dummy-client
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
