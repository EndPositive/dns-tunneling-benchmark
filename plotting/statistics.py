import json
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

def process_all_dfs(task_ids):
    all_dfs = []

    for task_id in task_ids:
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(f"../experiment/celery/artifacts/{task_id}.csv")

        # Ensure frame.time_relative is numeric (for grouping) and ip.len is in megabits (Mb)
        df['frame.time_relative'] = pd.to_numeric(df['frame.time_relative'], errors='coerce')
        df = df[df['frame.time_relative'] <= 90]

        # Round the frame.time_relative to nearest second for bucketing
        df['time_bin'] = df['frame.time_relative'].round()

        df['ip.len'] = pd.to_numeric(df['ip.len'], errors='coerce') * 8 / 1_000_000  # Convert to megabits

        # Group by time_bin, ip.dst, and interface_id, then sum the ip.len for each group
        df_grouped = df.groupby(['time_bin', 'ip.dst', 'frame.interface_id']).agg({'ip.len': 'sum'}).reset_index()

        dns_tunnel_server_ip = '172.22.0.6'
        iperf3_server_ip = '172.22.0.8'
        # whether this packet is upload or download
        df_grouped['type'] = df_grouped['ip.dst'].apply(lambda x: 'upload' if x == dns_tunnel_server_ip or x == iperf3_server_ip else 'download')

        # interface name
        df_grouped['interface'] = df_grouped['frame.interface_id'].apply(lambda x: 'outer' if x == 0 else 'inner')

        # Add a column to identify which task the data belongs to
        df_grouped['task_id'] = task_id

        all_dfs.append(df_grouped)

    return all_dfs


def plot_tasks(task_ids, all_dfs):
    # select task_ids from the all_dfs dataframe
    dfs = [df for df in all_dfs if df['task_id'].iloc[0] in task_ids]

    # Combine all task data into one DataFrame
    df_combined = pd.concat(dfs)

    # Create a line plot for the summed ip.len over time, with different traces per task
    fig = px.line(
        df_combined,
        x='time_bin',
        y='ip.len',
        color='type',
        line_dash='interface',
        symbol='task_id',
        title="Traffic over Time (Upload/Download per Interface per Task)",
        labels={
            'time_bin': 'Time (seconds)',
            'ip.len': 'Traffic Volume (Mb)',
            'color': 'Traffic Type',
            'line_dash': 'Interface',
            'task_id': 'Task ID'
        }
    )

    return fig

def get_celery_tasks():
    # Load JSON results from ../experiment/celery/results
    celery_task_results = []
    for child in Path('../experiment/celery/results').iterdir():
        if not child.is_file():
            continue

        child: Path = child

        result = json.load(child.open())

        celery_task_results.append(result)

    if len(celery_task_results) == 0:
        print("No celery tasks found")

    return celery_task_results


# Create initial figure with all selected traces
def create_figure(visible_tasks=None):
    if visible_tasks is None or len(visible_tasks) == 0:
        return None

    return plot_tasks(visible_tasks, all_dfs)


celery_task_results = get_celery_tasks()
task_ids = [celery_task_result["task_id"] for celery_task_result in celery_task_results]

all_dfs = process_all_dfs(task_ids)

app = Dash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(id="graph"),
        dcc.Dropdown(task_ids, id="task-id-selection", value=task_ids, multi=True),
    ]
)

# Dash callback to update the graph dynamically when tasks are selected
@app.callback(
    Output("graph", "figure"),
    Input("task-id-selection", "value"),
)
def modify_legend(task_id_selection):
    if task_id_selection is None or len(task_id_selection) == 0:
        task_id_selection = task_ids

    return create_figure(task_id_selection)

if __name__ == "__main__":
    app.run_server(debug=True)
