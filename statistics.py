import socket
from collections import defaultdict

import dpkt
from dash import Dash, Input, Output, dcc, html
import os
import plotly.graph_objs as go

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph"),
])


# Function to load pcap files and process the data
def load_pcap(file_name):
    ip_data = defaultdict(list)
    start_time = None

    with open(file_name, 'rb') as f:
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
    elif (ip_pair[1], ip_pair[0]) in ip_data:
        return ip_data[(ip_pair[1], ip_pair[0])]
    else:
        return []  # Return empty if no data found


# Load pcap files
pcap_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith('packets-') and f.endswith('.pcap')]
ip_datas = {}

for pcap_file in pcap_files:
    print("Loading", pcap_file)
    ip_data = load_pcap(pcap_file)
    ip_datas[pcap_file] = ip_data

# Now filter and plot for a specific IP pair
ip_pair = ('172.22.0.3', '172.22.0.2')
filtered_ip_data = {}
for pcap_file, ip_data in ip_datas.items():
    filtered_ip_data[pcap_file] = filter_ip_pair_data(ip_data, ip_pair)


# Function to compute dynamic bin size
def compute_bin_size_dynamic(visible_traces, protocol_packets, desired_bins=100):
    max_time = float('-inf')

    # Compute max time for visible traces
    for i, trace_visibility in enumerate(visible_traces):
        if trace_visibility:
            packet_time = list(protocol_packets.values())[i]
            max_time = max(max_time, max(packet_time))

    # Compute bin size
    bin_size = max_time / desired_bins if max_time > 0 else 1.0
    return bin_size, 0, max_time


# Create initial figure with all traces
def create_figure(visible_traces = None):
    fig = go.Figure()

    if not visible_traces:
        visible_traces = [True] * len(filtered_ip_data)

    # Compute dynamic bin size based on visible traces
    bin_size, min_time, max_time = compute_bin_size_dynamic(visible_traces, filtered_ip_data)

    # Add traces for each protocol
    for i, (protocol, packet_time) in enumerate(filtered_ip_data.items()):
        fig.add_trace(go.Histogram(
            x=packet_time,
            name=protocol,
            xbins=dict(start=min_time, size=bin_size, end=max_time),
            visible=visible_traces[i],
        ))

    # Update layout
    fig.update_layout(
        title="Packets per Second Histogram",
        xaxis_title="Time (seconds)",
        yaxis_title="Packet Count",
        showlegend=True,
        barmode='overlay',
    )

    return fig


# Dash callback to update the graph dynamically when legend is clicked
@app.callback(
    Output("graph", "figure"),
    Input("graph", "restyleData")  # Listen to changes in trace visibility
)
def modify_legend(restyleData):
    # If restyleData is empty, return the default figure
    if restyleData is None:
        return create_figure()

    edits, indices = restyleData

    # Extract trace visibility from restyleData
    if 'visible' in restyleData[0]:
        visible_traces = restyleData[0]['visible']
    else:
        visible_traces = None

    # Update the figure based on visible traces
    return create_figure(visible_traces)


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
