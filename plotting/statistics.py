import os
import socket
from collections import defaultdict

import dpkt
import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html


# Function to load pcap files and process the data
def load_pcap(file_name):
    ip_data = defaultdict(list)
    start_time = None

    with open(file_name, "rb") as f:
        pcap = dpkt.pcap.Reader(f)
        for timestamp, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)

                # Initialize start_time for the first packet
                if start_time is None:
                    start_time = timestamp

                time = timestamp - start_time

                ip_data[(src_ip, dst_ip)].append(time)

    return ip_data


def filter_ip_pair_data(ip_data, ip_pair):
    if ip_pair in ip_data:
        return ip_data[ip_pair]
    else:
        return []


# Load pcap files
pcap_files = [
    f"./results/{f}" for f in os.listdir("./results") if os.path.isfile(f"./results/{f}") and f.endswith(".pcap")
]
if len(pcap_files) == 0:
    print("No pcap files found in the results directory")
    exit(1)
ip_datas = {}

for pcap_file in pcap_files:
    print("Loading", pcap_file)
    ip_data = load_pcap(pcap_file)
    ip_datas[pcap_file] = ip_data

# Now filter and plot for a specific IP pair
filtered_ip_data = {}
for pcap_file, ip_data in ip_datas.items():
    filtered_ip_data[pcap_file] = filter_ip_pair_data(ip_data, ("172.22.0.3", "172.22.0.4"))
    if len(filtered_ip_data[pcap_file]) == 0:
        filtered_ip_data[pcap_file] = filter_ip_pair_data(ip_data, ("172.22.0.3", "172.22.0.5"))

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Graph(id="graph"),
        dcc.Dropdown(pcap_files, id="pcap-file-selection", value=pcap_files, multi=True),
    ]
)


# Create initial figure with all traces
def create_figure(visible_pcaps=[]):
    fig = go.Figure()

    # Add traces for each protocol
    for pcap_file, packet_time in filtered_ip_data.items():
        fig.add_trace(
            go.Histogram(
                x=packet_time,
                name=pcap_file,
                visible=True if pcap_file in visible_pcaps else "legendonly",
            )
        )

    # Update layout
    fig.update_layout(
        title="Packets per Second Histogram",
        xaxis_title="Time (seconds)",
        yaxis_title="Packet Count",
        showlegend=False,
        barmode="overlay",
    )

    return fig


# Dash callback to update the graph dynamically when legend is clicked
@app.callback(
    Output("graph", "figure"),
    Input("pcap-file-selection", "value"),
)
def modify_legend(pcap_file_selection):
    if pcap_file_selection is None:
        pcap_file_selection = pcap_files

    return create_figure(pcap_file_selection)


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
