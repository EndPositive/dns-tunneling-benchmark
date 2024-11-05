import json
import logging
from pathlib import Path

import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn

pd.options.plotting.backend = 'holoviews'

hv.extension('bokeh')

# Setup logger
logging.basicConfig(level=logging.INFO)

def process_all_dfs(celery_task_results):
    all_dfs = []
    global_max_time = 0  # To track the maximum time across all tasks

    for celery_task_result in celery_task_results:
        task_id = celery_task_result["task_id"]
        tunnel_name = celery_task_result["result"]["tunnel"]
        use_dns_resolver = celery_task_result["result"]["use_dns_resolver"]
        upload = celery_task_result["result"]["upload"]
        download = celery_task_result["result"]["download"]
        local = celery_task_result["result"]["local"]

        test_type = ""
        if upload and download:
            test_type = "bidir"
        elif upload:
            test_type = "upload"
        elif download:
            test_type = "download"

        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(f"../experiment/celery/artifacts/{task_id}.csv")

        # Convert columns to numeric, handle errors by coercing to NaN
        # Bin time to the nearest second
        df['frame.time_relative'] = np.ceil(pd.to_numeric(df['frame.time_relative'], errors='coerce'))
        df['frame.len'] = pd.to_numeric(df['frame.len'], errors='coerce')

        # Convert frame.len to megabits
        df['frame.len'] = df['frame.len'] * 8 / 1_000_000

        dns_tunnel_server_ip = '172.22.0.6'
        iperf3_server_ip = '172.22.0.8'
        iperf3_server_ip_public = '64.176.43.164'
        if tunnel_name.startswith("socks"):
            df['traffic_type'] = np.where(
                ((df['ip.dst'] == '127.0.0.1') & (df['tcp.dstport'] == 5201)) | ((df['ip.dst'] == '172.22.0.6') & (df['udp.dstport'] == 53)),
                'upload',
                'download'
            )
        elif tunnel_name == "raw":
            df['traffic_type'] = np.where(df['ip.dst'].isin([iperf3_server_ip, iperf3_server_ip_public]), 'upload', 'download')
        else:
            df['traffic_type'] = np.where(df['ip.dst'].isin([dns_tunnel_server_ip, iperf3_server_ip, iperf3_server_ip_public]), 'upload', 'download')

        # Convert frame.len to negative if 'upload'
        df['frame.len'] = np.where(df['traffic_type'] == 'upload', -df['frame.len'], df['frame.len'])

        # interface_id: 0 (eth0) = inner
        # interface_id: 1 (lo) = outer
        # interface_id: 2 (dns0) = outer
        df['interface'] = np.where(df['frame.interface_id'] == 0, 'inner', 'outer')

        # Group by 'frame.time_relative', 'traffic_type', 'interface' and sum frame.len
        df_grouped = df.groupby(['frame.time_relative', 'traffic_type', 'interface'], as_index=False)['frame.len'].sum()

        if tunnel_name == "raw":
            # Duplicate the grouped rows where the interface is 'inner'
            df_inner = df_grouped[df_grouped['interface'] == 'inner'].copy()
            df_inner['interface'] = 'outer'  # Change 'inner' to 'outer'

            # Append the duplicated outer rows to the grouped dataframe
            df_grouped = pd.concat([df_grouped, df_inner], ignore_index=True)

        # clear df to save memory
        del df

        # Update the global maximum time
        task_max_time = df_grouped['frame.time_relative'].max()
        global_max_time = max(global_max_time, task_max_time)

        # Store the grouped DataFrame for later concatenation
        all_dfs.append((df_grouped, task_id, tunnel_name, use_dns_resolver, test_type, local))

    # After processing all tasks, create the complete index based on the global max_time
    complete_index = pd.MultiIndex.from_product(
        [np.arange(0, global_max_time + 1), ['upload', 'download'], ['inner', 'outer']],
        names=['frame.time_relative', 'traffic_type', 'interface']
    )

    # Process all stored DataFrames and reindex them
    final_dfs = []
    for df_grouped, task_id, tunnel_name, use_dns_resolver, test_type, local in all_dfs:
        df_grouped = df_grouped.set_index(['frame.time_relative', 'traffic_type', 'interface']).reindex(complete_index, fill_value=0).reset_index()

        # Add interface, task_id, tunnel_name, and use_dns_resolver columns
        df_grouped['task_id'] = task_id
        df_grouped['tunnel_name'] = tunnel_name
        df_grouped['use_dns_resolver'] = use_dns_resolver
        df_grouped['test_type'] = test_type
        df_grouped['local'] = local

        final_dfs.append(df_grouped)

    return pd.concat(final_dfs)


def get_celery_tasks():
    celery_task_results = []
    for child in Path('../experiment/celery/results').iterdir():
        if not child.is_file():
            continue
        result = json.load(child.open())
        celery_task_results.append(result)
    return celery_task_results

celery_task_results = get_celery_tasks()
all_dfs = process_all_dfs(celery_task_results)
task_ids = [celery_task_result["task_id"] for celery_task_result in celery_task_results]
tunnel_names = list(set([celery_task_result["result"]["tunnel"] for celery_task_result in celery_task_results]))


task_ids_select = pn.widgets.MultiSelect(
    name='Task ID',
    options=task_ids,
    value=task_ids
)
tunnel_names_select = pn.widgets.MultiSelect(
    name='Tunnel Name',
    options=tunnel_names,
    value=tunnel_names
)
interfaces_select = pn.widgets.MultiSelect(
    name='Interface',
    options=['outer', 'inner'],
    value=['outer', 'inner']
)
traffic_types_select = pn.widgets.MultiSelect(
    name='Trafic Type',
    options=['upload', 'download'],
    value=['upload', 'download']
)
test_types_select = pn.widgets.MultiSelect(
    name='Test Type',
    options=['upload', 'download', 'bidir'],
    value=['upload', 'download', 'bidir']
)
local_select = pn.widgets.MultiSelect(
    name='Local',
    options=[True, False],
    value=[True, False]
)

@pn.depends(task_ids_select.param.value, tunnel_names_select.param.value, test_types_select.param.value, traffic_types_select.param.value, interfaces_select.param.value, local_select.param.value)
def plot(task_ids, tunnel_names, test_types, types, interfaces, local):
    df_filtered = all_dfs[
        all_dfs['task_id'].isin(task_ids) &
        all_dfs['tunnel_name'].isin(tunnel_names) &
        all_dfs['test_type'].isin(test_types) &
        all_dfs['traffic_type'].isin(types) &
        all_dfs['interface'].isin(interfaces) &
        all_dfs['local'].isin(local)
    ]

    if df_filtered.empty:
        # TODO: return a message instead of an empty plot
        return

    df_filtered = df_filtered.sort_values(by='frame.time_relative')

    return df_filtered.plot.line(
        x='frame.time_relative',
        y='frame.len',
        by=['tunnel_name', 'traffic_type', 'interface', 'task_id', 'test_type'],
        ylabel='Sum of frame.len (Mbit)',
        xlabel='Time (relative)',
        responsive=True,
        height=800,
        legend='bottom',
    )

dashboard = pn.Column(
    pn.Row(task_ids_select, tunnel_names_select, test_types_select, traffic_types_select, interfaces_select, local_select),
    plot
)

if __name__ == "__main__":
    pn.serve(dashboard, start=True, port=5006)

