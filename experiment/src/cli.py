# fmt: off
import json
import math
import sys
from typing import Annotated, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2
import typer
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
from python_on_whales import DockerClient

from .lib.docker import new_docker_controller, new_local_dns_resolvers_docker_controller
from .lib.lib import Env, Tunnel
from .lib.log import log_stream_logs, logger, safe_log
from .tasks.run import NetEmClient, NetEmServer
from .lib.tests import Test

cli = typer.Typer()
opt_verbose = False
opt_client_docker_host = None
opt_server_docker_host = None

opt_type_test = typer.Option(case_sensitive=False, rich_help_panel="Experiment configuration")
opt_dry_run = typer.Option(case_sensitive=False, rich_help_panel="Experiment configuration")
opt_type_tunnel = typer.Option(case_sensitive=True, rich_help_panel="Experiment configuration")
opt_type_dns_resolver = Annotated[List[str], typer.Option(help="DNS resolver", rich_help_panel="Experiment configuration")]
opt_type_dns_resolver_rate_limit = Annotated[int, typer.Option(help="Rate limit for DNS resolver (manually set; just for administration)", rich_help_panel="Experiment configuration")]
opt_type_net_em_client = typer.Option(
    case_sensitive=False,
    help="\n".join(f"{name}: {NetEmClient[name].config}" for name in NetEmClient.__members__),
    rich_help_panel="Network emulation",
)
opt_type_net_em_server = typer.Option(
    case_sensitive=False,
    help="\n".join(f"{name}: {NetEmServer[name].config}" for name in NetEmServer.__members__),
    rich_help_panel="Network emulation",
)
opt_type_website_path = typer.Option(
    case_sensitive=True,
    help="path to the website to test (only applicable for browsing test)",
    rich_help_panel="Browsing test configuration",
)
opt_type_db = typer.Option(help="PostgreSQL connection string", rich_help_panel="Database")
opt_type_db_default = "postgresql://username:password@localhost:5432/postgres"
opt_type_docker_host = typer.Option(help="Client-side Docker host (e.g. tcp://1.2.3.4:2376", rich_help_panel="Docker hosts")


def sizeof_fmt(num, suffix='ps', bit=False):
    # Check if the number is negative
    sign = '-' if num < 0 else ''
    num = abs(num)  # Work with the absolute value for formatting
    if bit:
        num *= 8

    bit_byte = "B"
    if bit:
        bit_byte = "b"
    for x in [f'{bit_byte}', f'K{bit_byte}', f'M{bit_byte}', f'G{bit_byte}', f'T{bit_byte}']:
        if num < 1000.0:
            if num == 0:
                format_spec = ".1f"  # Keep 0.0 format
            elif num >= 100:
                format_spec = ".0f"  # e.g., 123
            elif num >= 10:
                format_spec = ".1f"  # e.g., 12.3
            else:  # 0 < num < 10
                format_spec = ".2f"  # e.g., 1.23 or 0.12
            return f"{sign}{num:{format_spec}} {x}{suffix}"
        num /= 1000.0

    return f"{sign}{num:.1f} T{bit_byte}{suffix}"  # Handle very large numbers
formatter_ps = FuncFormatter(lambda x, _: sizeof_fmt(x))
formatter = FuncFormatter(lambda x, _: sizeof_fmt(x, ""))


@cli.command()
def plot_browsing_test(
        resolver_count: Annotated[int, typer.Option(help="Number of DNS resolvers", rich_help_panel="Experiment configuration")] = 1,
        resolver_rate_limit: Annotated[int, opt_type_dns_resolver_rate_limit] = 0,
        db: Annotated[str, opt_type_db] = opt_type_db_default,
):
    connection = psycopg2.connect(db)

    with connection.cursor() as cursor:
        cursor.execute("""
                        SELECT 
                            tm.task_id,
                            tm.tunnel_name,
                            tm.website_path,
                            ROUND(fd.avg/1000, 0) as avg,
                            ROUND(fd.min/1000, 0) as min,
                            ROUND(fd.med/1000, 0) as med,
                            ROUND(fd.avg/1000, 0) as max,
                            fd.p90,
                            fd.p95,
                            um.total_queries
                        FROM browsing_test_results fd
                        JOIN task_metadata tm ON fd.task_id::text = tm.task_id::text
                        LEFT JOIN (
                            SELECT 
                                task_id,
                                SUM(value) AS total_queries
                            FROM unbound_metrics
                            WHERE metric = 'total_num_queries'
                            GROUP BY task_id
                        ) um ON um.task_id = fd.task_id
                        WHERE
                            tm.test_type = 'browsing'
                            AND jsonb_array_length(tm.dns_resolvers) = %s
                            AND tm.dns_resolver_rate_limit = %s
                        ORDER BY
                            tm.tunnel_name DESC;
                       """, (resolver_count, resolver_rate_limit))

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    # check if there are any duplicate tm.tunnel_name, tm.website_path
    if df.duplicated(subset=['tunnel_name', 'website_path']).any():
        raise Exception("make sure to have no duplicates test results")

    unique_websites = df['website_path'].unique()
    n_websites = len(unique_websites)

    # Set seaborn style for consistency with other plots
    sns.set(style='whitegrid')

    fig_width = 8
    fig_height = 3
    
    fig, axes = plt.subplots(1, n_websites, figsize=(fig_width, fig_height), dpi=300, squeeze=False)

    # --- Calculate Y-axis limits ---
    min_val_ms = float(df['min'].min())
    max_val_ms = float(df['max'].max())
    data_range_ms = max_val_ms - min_val_ms
    padding_ms = data_range_ms * 0.15  # Add 15% padding above and below for better visibility
    lower_limit_ms = math.floor(max(0, min_val_ms - padding_ms))  # Ensure lower limit is not negative
    upper_limit_ms = math.ceil(max_val_ms + padding_ms)

    available_websites = [
        "news.ycombinator.org/news.ycombinator.org",
        "esorics2025.sciencesconf.org/esorics2025.sciencesconf.org",
        "cve.mitre.org/cve.mitre.org",
        "owasp.org/owasp.org"
    ]
    names = ["A", "B", "C", "D"]

    for i, website in enumerate(available_websites):
        if website not in unique_websites:
            print(f"missing {website}")
            continue

        ax1 = axes[0, i]
        df_website = df[df['website_path'] == website].copy()
        
        # Format website name for display
        website_display = names[i]

        # Plot average bars
        sns.barplot(
            data=df_website, 
            x='tunnel_name', 
            y='avg', 
            color='#66C2FF', 
            label='Average',
            ax=ax1
        )

        # Calculate errors for min/max relative to the average
        lower_error = df_website['avg'] - df_website['min']
        upper_error = df_website['max'] - df_website['avg']
        errors = [lower_error.tolist(), upper_error.tolist()]

        # Get x-coordinates for the bars
        x_coords = np.arange(len(df_website))

        # Add Min/Max error bars
        ax1.errorbar(
            x=x_coords, 
            y=df_website['avg'], 
            yerr=errors, 
            fmt='none',
            capsize=10,
            color='black', 
            elinewidth=2,
            label='Min/Max Range'
        )

        # Add markers for Median, P90, P95
        ax1.scatter(x=x_coords, y=df_website['med'], color='red', marker='_', s=150, label='Median', zorder=3)

        # --- Formatting ---
        ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # Set title and labels
        ax1.set_title(website_display, fontsize=12, pad=10)
        ax1.set_xlabel('')
        
        if i == 0:
            ax1.set_ylabel('Time (s.)', fontsize=10)
        else:
            ax1.set_ylabel('')
            ax1.set_yticklabels([])

        # set legend
        if i == 0:
            ax1.legend(
                loc='upper right',
                fontsize=8,
                title='Metrics',
                title_fontsize=9,
                frameon=True,
                facecolor='white',
                edgecolor='black'
            )
        else:
            ax1.legend().remove()

        # Set y-axis limits
        ax1.set_ylim(bottom=lower_limit_ms, top=upper_limit_ms)
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Format x-tick labels
        tunnel_names = [name.split('/')[-1] for name in df_website['tunnel_name']]
        ax1.set_xticklabels(tunnel_names, rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    plt.show()

@cli.command()
def plot_file_transfer(
        resolver_count: Annotated[int, typer.Option(help="Number of DNS resolvers", rich_help_panel="Experiment configuration")] = 0,
        resolver_rate_limit: Annotated[int, opt_type_dns_resolver_rate_limit] = 0,
        upload: Annotated[bool, typer.Option(help="Upload test", rich_help_panel="Experiment configuration")] = False,
        db: Annotated[str, opt_type_db] = opt_type_db_default,
):
    connection = psycopg2.connect(db)

    test_type = "file-download"
    if upload:
        test_type = "file-upload"

    with connection.cursor() as cursor:
        cursor.execute("""
                       SELECT tm.task_id,
                              tm.tunnel_name,
                              fd.event_type,
                              fd.time,
                              CASE
                                WHEN fd.event_type = 'read' THEN SUM(fd.bytes_received) OVER (PARTITION BY fd.task_id ORDER BY fd.time)
                                WHEN fd.event_type IN ('total_num_queries', 'total_num_queries_ip_ratelimited') THEN fd.bytes_received
                                ELSE -1
                              END AS cumulative_bytes_received
                       FROM file_download_result fd
                                JOIN
                            task_metadata tm ON fd.task_id::text = tm.task_id::text
                       WHERE
                           (fd.event_type = 'read' OR fd.event_type = 'total_num_queries' OR fd.event_type = 'total_num_queries_ip_ratelimited') AND
                           -- (starts_with(tm.tunnel_name, 'tcp/slipstream') OR tm.tunnel_name = 'tcp/dnstt') AND
                           tm.test_type = %s AND
                           jsonb_array_length(tm.dns_resolvers) = %s AND
                           tm.dns_resolver_rate_limit = %s
                         AND (fd.error = '' or fd.error IS NULL)
                       ORDER BY
                           tm.task_id, fd.time;
                       """, (test_type, resolver_count, resolver_rate_limit))

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    # exclude rows with event_type 'total_num_queries' or 'total_num_queries_ip_ratelimited' (copy those to seperate df)
    unbound_stats = df[df['event_type'].isin(['total_num_queries', 'total_num_queries_ip_ratelimited'])]
    # rename columns to task_id, tunnel_name, time->resolver_id, event_type->metric_type, drop, cumulative_bytes_received->metric_name
    unbound_stats = unbound_stats.rename(columns={
        'event_type': 'metric_type',
        'time': 'resolver_id',
        'cumulative_bytes_received': 'metric_name'
    })

    # drop event_type column
    df = df[~df['event_type'].isin(['total_num_queries', 'total_num_queries_ip_ratelimited'])]
    # drop event_type column
    df = df.drop(columns=['event_type'])

    # convert time to seconds
    df['time'] = df['time'].astype(np.int64) / 1e9
    df['time_round'] = df['time'].round(0).astype(int)
    # Aggregate data by task_id and time_round, keeping the last cumulative value for each second
    # Ensure tunnel_name is carried through the grouping
    df = df.groupby(['task_id', 'tunnel_name', 'time_round']).last().reset_index()

    # find the execution that has the lowest finishing time for each tunnel_name
    # 1. Find the maximum time (finish time) for each task_id
    finishing_times = df.loc[df.groupby('task_id')['time'].idxmax()]

    # 2. Find the task_id with the minimum finish time for each tunnel_name
    fastest_task_indices = finishing_times.loc[finishing_times.groupby('tunnel_name')['time'].idxmin()]
    fastest_task_ids = fastest_task_indices['task_id'].unique()

    # 3. Filter the original DataFrame to keep only the data from the fastest tasks
    df = df[df['task_id'].isin(fastest_task_ids)].copy()

    # Calculate final stats and average speed per tunnel
    last_rows_idx = df.groupby('task_id')['time'].idxmax()
    avg_speed_per_tunnel = df.loc[last_rows_idx]\
        .rename(columns={'time': 'duration', 'cumulative_bytes_received': 'total_bytes'})\
        .assign(
            total_bytes=lambda x: x['total_bytes'].astype(float),
            speed_bps=lambda x: np.where(x['duration'] > 0, x['total_bytes'] / x['duration'], 0.0),
        )\
        .groupby('tunnel_name')[['speed_bps', 'duration']]\
        .mean()
    original = avg_speed_per_tunnel.to_dict()

    # Collect all unique protocols from all metrics
    protocols = set()
    for inner_dict in original.values():
        protocols.update(inner_dict.keys())

    # Build the restructured dictionary
    result = {}
    for proto in protocols:
        proto_entry = {}
        for metric in original:
            if proto in original[metric]:
                proto_entry[metric] = original[metric][proto]
        result[proto] = proto_entry

    print("tunnel_name,speed,speed_str,duration")
    for tunnel_name, stats in result.items():
        speed = stats['speed_bps']
        speed_str = sizeof_fmt(speed, suffix='ps', bit=True)
        duration = stats['duration']
        print(f"{tunnel_name},{speed},{speed_str},{duration}")
    print("")

    unbound_stats = unbound_stats[unbound_stats['task_id'].isin(fastest_task_ids)].copy()
    unbound_stats = unbound_stats.drop(columns=['task_id'])
    unbound_stats.to_csv(sys.stdout, index=False)

    custom_order = ["tcp/slipstream", "tcp/dnstt", "fd/dns-tunnler", "tun/iodine", "tun/TUNS", "fd/OzymanDNS", "fd/sods", "tcp/dns2tcp"]

    plt.figure(figsize=(8, 3), dpi=300, tight_layout=True)
    for tunnel_name in custom_order:
        if tunnel_name not in df['tunnel_name'].unique():
            continue
        subset = df[df['tunnel_name'] == tunnel_name]

        sns.lineplot(
            x='time',
            y='cumulative_bytes_received',
            data=subset,
            label=tunnel_name,
            alpha=0.7,
            linewidth=3,
            palette='colorblind',
        )

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().yaxis.set_major_formatter(formatter)
    # plt.xlim(left=0, right=300)
    plt.ylim(bottom=0, top=1e7)
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative bytes transferred')

    # Simple legend without speed information
    plt.legend(title='Tunnel', loc='center right', frameon=True)

    plt.show()

@cli.command()
def plot_resolver_comparison(
        db: Annotated[str, opt_type_db] = opt_type_db_default,
):
    connection = psycopg2.connect(db)

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT
                tm.task_id,
                tm.tunnel_name,
                tm.test_type,
                tm.netem_client_name,
                tm.netem_server_name,
                tm.dns_resolvers,
                SUM(pd.len_upload_perceived_server) / 15 AS upload_perceived_server,
                SUM(pd.len_upload_actual_client) / 15 AS upload_actual_client,
                -SUM(pd.len_download_perceived_client) / 15 AS download_perceived_client,
                -SUM(pd.len_download_actual_client) / 15 AS download_actual_client,
                COUNT(CASE WHEN pd.len_upload_actual_client > 0 THEN 1 END) / 15 AS upload_actual_client_count,
                COUNT(CASE WHEN pd.len_download_actual_client > 0 THEN 1 END) / 15 AS download_actual_client_count,
                jsonb_array_length(tm.dns_resolvers) AS dns_resolvers_count
            FROM
                packet_data pd
                    JOIN
                task_metadata tm ON pd.task_id::text = tm.task_id::text
            WHERE
                pd.frame_time_floor >= 25 AND pd.frame_time_floor <= 40 AND
                tm.test_type IN ('upload', 'download', 'bidir') AND
                tm.netem_client_name = 'university' AND
                tm.netem_server_name = 'international' AND
                tm.tunnel_name = 'tcp/slipstream' AND
                jsonb_array_length(tm.dns_resolvers) > 0
            GROUP BY
                tm.task_id, tm.tunnel_name, tm.test_type, tm.netem_client_name, tm.netem_server_name, tm.dns_resolvers;
        """)

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    # Set the style of seaborn
    sns.set(style='whitegrid')

    # Create a bar chart for each test_type
    test_types = ["upload", "download", "bidir"]

    plt.figure(figsize=(10, 6), dpi=300)

    # everything in one big figure (use index in range)
    for i, test in enumerate(test_types):
        plt.subplot(2, len(test_types), i+1)
        subset = df[df['test_type'] == test]

        # upload
        subset_melted = subset.melt(
            id_vars=['dns_resolvers_count'],
            value_vars=['upload_actual_client', 'upload_perceived_server'],
            var_name='Type',
            value_name='Bandwidth'
        )
        subset_melted['Type'] = subset_melted['Type'].map({
            'upload_actual_client': 'Overhead',
            'upload_perceived_server': 'Goodput'
        })
        sns.barplot(x='dns_resolvers_count',
                    y='Bandwidth',
                    hue='Type',
                    data=subset_melted,
                    dodge=False,
                    palette={'Overhead': 'lightgrey', 'Goodput': '#66C266'},
                    )

        # download
        subset_melted = subset.melt(
            id_vars=['dns_resolvers_count'],
            value_vars=['download_actual_client', 'download_perceived_client'],
            var_name='Type',
            value_name='Bandwidth'
        )
        subset_melted['Type'] = subset_melted['Type'].map({
            'download_actual_client': 'Overhead',
            'download_perceived_client': 'Goodput'
        })
        sns.barplot(x='dns_resolvers_count',
                    y='Bandwidth',
                    hue='Type',
                    data=subset_melted,
                    dodge=False,
                    palette={'Overhead': 'lightgrey', 'Goodput': '#66C266'},
                    legend=False,
                    )

        plt.axhline(0, color='black', linewidth=2)

        plt.title(f'{test.capitalize()} test')
        plt.ylim(-3e6, 8e5)
        plt.xlabel('')
        if i != 0:
            plt.ylabel('')
        plt.gca().yaxis.set_major_formatter(formatter_ps)
        plt.gca().set_xticklabels([])
        if i == len(test_types) - 1:
            plt.legend(loc='center right')
        else:
            plt.legend().remove()
        # plt.yscale('symlog', linthresh=10e4)


        plt.subplot(2, len(test_types), (i+4))
        subset_melted = subset.melt(
            id_vars=['dns_resolvers_count'],
            value_vars=['upload_actual_client_count', 'download_actual_client_count'],
            var_name='Type',
            value_name='Queries per second'
        )
        subset_melted['Type'] = subset_melted['Type'].map({
            'upload_actual_client_count': 'Lost',
            'download_actual_client_count': 'Received'
        })

        sns.barplot(
            x='dns_resolvers_count',
            y='Queries per second',
            hue='Type',
            data=subset_melted,
            dodge=False,
            palette={'Lost': 'lightgrey', 'Received': '#66C266'},
        )
        plt.ylim(0, 3500)
        if i == len(test_types) - 1:
            plt.legend(loc='lower right')
        else:
            plt.legend().remove()
        plt.xticks(rotation=45, ha='right')
        if i == 1:
            plt.xlabel('Number of DNS Resolvers')
        else:
            plt.xlabel('')
        if i != 0:
            plt.ylabel('')

    plt.tight_layout()
    plt.show()

@cli.command()
def plot_network_comparison(
        tunnel: Annotated[Tunnel, opt_type_tunnel],
        db: Annotated[str, opt_type_db] = opt_type_db_default,
):
    connection = psycopg2.connect(db)

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT
                tm.task_id,
                tm.tunnel_name,
                tm.test_type,
                tm.netem_client_name,
                tm.netem_server_name,
                tm.dns_resolvers,
                SUM(pd.len_upload_perceived_server) / 30 AS upload_perceived_server,
                SUM(pd.len_upload_actual_client) / 30 AS upload_actual_client,
                -SUM(pd.len_download_perceived_client) / 30 AS download_perceived_client,
                -SUM(pd.len_download_actual_client) / 30 AS download_actual_client,
                COUNT(CASE WHEN pd.len_upload_actual_client > 0 THEN 1 END) / 30 AS upload_actual_client_count,
                COUNT(CASE WHEN pd.len_download_actual_client > 0 THEN 1 END) / 30 AS download_actual_client_count,
                CASE WHEN tm.netem_client_name = 'university' AND tm.netem_server_name = 'international' THEN 'S1'
                        WHEN tm.netem_client_name = 'university' AND tm.netem_server_name = 'national' THEN 'S2'
                        WHEN tm.netem_client_name = 'mobile' AND tm.netem_server_name = 'international' THEN 'S3'
                        WHEN tm.netem_client_name = 'mobile' AND tm.netem_server_name = 'national' THEN 'S4'
                END AS scenario
            FROM
                packet_data pd
                    JOIN
                task_metadata tm ON pd.task_id::text = tm.task_id::text
            WHERE
                pd.frame_time_floor >= 10 AND pd.frame_time_floor <= 40 AND
                tm.test_type IN ('upload', 'download', 'bidir') AND
                tm.tunnel_name = %s AND
                tm.dns_resolvers = '["local"]'
            GROUP BY
                tm.task_id, tm.tunnel_name, tm.test_type, tm.netem_client_name, tm.netem_server_name, tm.dns_resolvers;
        """, (tunnel.value, ))

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    custom_order = ["S1", "S2", "S3", "S4"]

    # Set the style of seaborn
    sns.set(style='whitegrid')

    # Create a bar chart for each test_type
    test_types = ["upload", "download", "bidir"]

    plt.figure(figsize=(10, 3), dpi=300)

    # everything in one big figure (use index in range)
    for i, test in enumerate(test_types):
        plt.subplot(1, len(test_types), i+1)
        subset = df[df['test_type'] == test]

        # upload
        subset_melted = subset.melt(
            id_vars=['scenario'],
            value_vars=['upload_actual_client', 'upload_perceived_server'],
            var_name='Type',
            value_name='Bandwidth'
        )
        subset_melted['Type'] = subset_melted['Type'].map({
            'upload_actual_client': 'Overhead',
            'upload_perceived_server': 'Goodput'
        })
        sns.barplot(x='scenario',
                    y='Bandwidth',
                    hue='Type',
                    data=subset_melted,
                    dodge=False,
                    palette={'Overhead': 'lightgrey', 'Goodput': '#66C266'},
                    order=custom_order)

        # download
        subset_melted = subset.melt(
            id_vars=['scenario'],
            value_vars=['download_actual_client', 'download_perceived_client'],
            var_name='Type',
            value_name='Bandwidth'
        )
        subset_melted['Type'] = subset_melted['Type'].map({
            'download_actual_client': 'Overhead',
            'download_perceived_client': 'Goodput'
        })
        sns.barplot(x='scenario',
                    y='Bandwidth',
                    hue='Type',
                    data=subset_melted,
                    dodge=False,
                    palette={'Overhead': 'lightgrey', 'Goodput': '#66C266'},
                    legend=False,
                    order=custom_order)

        plt.axhline(0, color='black', linewidth=2)

        plt.title(f'{test.capitalize()} test')
        plt.ylim(-1e6, 3e5)
        plt.xlabel('')
        if i != 0:
            plt.ylabel('')
        plt.gca().yaxis.set_major_formatter(formatter_ps)
        if i == len(test_types) - 1:
            plt.legend(loc='center right')
        else:
            plt.legend().remove()

        plt.xticks(rotation=45, ha='right')
        if i == 1:
            plt.xlabel('Network Emulation')
        else:
            plt.xlabel('')
        if i != 0:
            plt.ylabel('')
            plt.yticks([])

    plt.tight_layout()
    plt.show()


@cli.command()
def pivot_tool_comparison(
        dns_resolver: Annotated[List[str], opt_type_dns_resolver] = [],
        db: Annotated[str, opt_type_db] = opt_type_db_default,
):
    connection = psycopg2.connect(db)

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT
                tm.task_id,
                tm.tunnel_name,
                tm.test_type,
                tm.netem_client_name,
                tm.netem_server_name,
                tm.dns_resolvers,
                SUM(pd.len_upload_perceived_server) / 30 * 8 / 1024 AS upload_perceived_server,
                SUM(pd.len_upload_actual_client) / 30 * 8 / 1024 AS upload_actual_client,
                SUM(pd.len_download_perceived_client) / 30 * 8 / 1024 AS download_perceived_client,
                SUM(pd.len_download_actual_client) / 30 * 8 / 1024 AS download_actual_client,
                COUNT(CASE WHEN pd.len_upload_actual_client > 0 THEN 1 END) / 30 AS upload_actual_client_count,
                COUNT(CASE WHEN pd.len_download_actual_client > 0 THEN 1 END) / 30 AS download_actual_client_count
            FROM
                packet_data pd
                    JOIN
                task_metadata tm ON pd.task_id::text = tm.task_id::text
            WHERE
                pd.frame_time_floor >= 10 AND pd.frame_time_floor <= 40 AND
                tm.test_type IN ('upload', 'download', 'bidir') AND
                tm.netem_client_name = 'university' AND
                tm.netem_server_name = 'international' AND
                tm.dns_resolvers = %s
            GROUP BY
                tm.task_id, tm.tunnel_name, tm.test_type, tm.netem_client_name, tm.netem_server_name, tm.dns_resolvers;
        """, (json.dumps(dns_resolver), ))

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    df = df[~((df['test_type'] == 'upload') & (df['upload_perceived_server'] == 0))]
    df = df[~((df['test_type'] == 'download') & (df['download_perceived_client'] == 0))]
    df = df[~((df['test_type'] == 'bidir') & (df['upload_perceived_server'] == 0) & (df['download_perceived_client'] == 0))]

    # sum all perceived and minus the actual
    df['overhead'] = df['upload_actual_client'] + df['download_actual_client'] - df['upload_perceived_server'] - df['download_perceived_client']
    df['overhead_ratio'] = df['overhead'] / (df['upload_perceived_server'] + df['download_perceived_client'])
    df['loss_rate'] = df['download_actual_client_count'] - df['upload_actual_client_count']
    df['loss_rate'] = df['loss_rate'].clip(lower=0)

    custom_order = [tunnel.value for tunnel in Tunnel if tunnel.value in df['tunnel_name'].unique()]
    df['tunnel_name'] = df['tunnel_name'].str.split('/').str[1]
    custom_order = [tunnel.split('/')[1] for tunnel in custom_order]

    pivot_df = df.pivot_table(
        index=['tunnel_name'],
        columns=['test_type'],
        values=['upload_perceived_server', 'download_perceived_client',
                'overhead_ratio', 'download_actual_client_count', 'loss_rate']
    )

    pivot_df = pivot_df.fillna(0)
    for col in pivot_df.columns:
        if col[0] != 'overhead_ratio':
            pivot_df[col] = pivot_df[col].astype(int)
        else:
            pivot_df[col] = pivot_df[col].apply(lambda x: f'{x:.0%}').apply(lambda x: x.replace('%', ''))

    # Custom column order
    desired_metrics = [
        'upload_perceived_server', 'download_perceived_client',
        'overhead_ratio', 'download_actual_client_count',
        'loss_rate'
    ]

    # Manual test_type order
    manual_test_type_order = ['upload', 'download', 'bidir']

    # Reorder columns to follow manual test_type and metric order
    pivot_df = pivot_df.reorder_levels([1, 0], axis=1)  # Reorder to (test_type, metric)
    pivot_df = pivot_df.loc[:, pd.MultiIndex.from_product(
        [manual_test_type_order, desired_metrics],
        names=pivot_df.columns.names
    )]


    pivot_df = pivot_df.reindex(custom_order)

    pivot_df.reset_index(inplace=True)

    pivot_df.drop(columns=[('upload', 'download_perceived_client')], inplace=True)
    pivot_df.drop(columns=[('download', 'upload_perceived_server')], inplace=True)

    # save as CSV (without any headers)
    pivot_df.to_csv('pivot_tool_comparison.csv', index=False)

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-detect the display width
    pd.set_option('display.max_colwidth', None)  # Show full content of each column

    print(pivot_df)

@cli.command()
def plot_tool_comparison(
        dns_resolver: Annotated[List[str], opt_type_dns_resolver] = [],
        db: Annotated[str, opt_type_db] = opt_type_db_default,
):
    connection = psycopg2.connect(db)

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT
                tm.task_id,
                tm.tunnel_name,
                tm.test_type,
                tm.netem_client_name,
                tm.netem_server_name,
                tm.dns_resolvers,
                SUM(pd.len_upload_perceived_server) / 15 AS upload_perceived_server,
                SUM(pd.len_upload_actual_client) / 15 AS upload_actual_client,
                -SUM(pd.len_download_perceived_client) / 15 AS download_perceived_client,
                -SUM(pd.len_download_actual_client) / 15 AS download_actual_client,
                COUNT(CASE WHEN pd.len_upload_actual_client > 0 THEN 1 END) / 15 AS upload_actual_client_count,
                COUNT(CASE WHEN pd.len_download_actual_client > 0 THEN 1 END) / 15 AS download_actual_client_count
            FROM
                packet_data pd
                    JOIN
                task_metadata tm ON pd.task_id::text = tm.task_id::text
            WHERE
                pd.frame_time_floor >= 25 AND pd.frame_time_floor <= 40 AND
                tm.test_type IN ('upload', 'download', 'bidir') AND
                tm.netem_client_name = 'university' AND
                tm.netem_server_name = 'international' AND
                tm.dns_resolvers = %s
            GROUP BY
                tm.task_id, tm.tunnel_name, tm.test_type, tm.netem_client_name, tm.netem_server_name, tm.dns_resolvers;
        """, (json.dumps(dns_resolver), ))

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    custom_order = [tunnel.value for tunnel in Tunnel if tunnel.value in df['tunnel_name'].unique()]
    df['tunnel_name'] = df['tunnel_name'].str.split('/').str[1]
    custom_order = [tunnel.split('/')[1] for tunnel in custom_order]


    # Set the style of seaborn
    sns.set(style='whitegrid')

    # Create a bar chart for each test_type
    test_types = ["upload", "download", "bidir"]

    plt.figure(figsize=(10, 6), dpi=300)

    # everything in one big figure (use index in range)
    for i, test in enumerate(test_types):
        plt.subplot(2, len(test_types), i+1)
        subset = df[df['test_type'] == test]

        # upload
        subset_melted = subset.melt(
            id_vars=['tunnel_name'],
            value_vars=['upload_actual_client', 'upload_perceived_server'],
            var_name='Type',
            value_name='Bandwidth'
        )
        subset_melted['Type'] = subset_melted['Type'].map({
            'upload_actual_client': 'Overhead',
            'upload_perceived_server': 'Goodput'
        })
        sns.barplot(x='tunnel_name',
                    y='Bandwidth',
                    hue='Type',
                    data=subset_melted,
                    dodge=False,
                    palette={'Overhead': 'lightgrey', 'Goodput': '#66C266'},
                    order=custom_order)

        # download
        subset_melted = subset.melt(
            id_vars=['tunnel_name'],
            value_vars=['download_actual_client', 'download_perceived_client'],
            var_name='Type',
            value_name='Bandwidth'
        )
        subset_melted['Type'] = subset_melted['Type'].map({
            'download_actual_client': 'Overhead',
            'download_perceived_client': 'Goodput'
        })
        sns.barplot(x='tunnel_name',
                    y='Bandwidth',
                    hue='Type',
                    data=subset_melted,
                    dodge=False,
                    palette={'Overhead': 'lightgrey', 'Goodput': '#66C266'},
                    legend=False,
                    order=custom_order)

        plt.axhline(0, color='black', linewidth=2)

        plt.title(f'{test.capitalize()} test')
        if len(dns_resolver) == 0:
            plt.ylim(-1e7, 8e6)
        else:
            plt.ylim(-1e6, 3e5)
        plt.xlabel('')
        if i != 0:
            plt.ylabel('')
        plt.gca().yaxis.set_major_formatter(formatter_ps)
        plt.gca().set_xticklabels([])
        if i == len(test_types) - 1:
            plt.legend(loc='lower right')
        else:
            plt.legend().remove()

        plt.subplot(2, len(test_types), (i+4))
        subset_melted = subset.melt(
            id_vars=['tunnel_name'],
            value_vars=['upload_actual_client_count', 'download_actual_client_count'],
            var_name='Type',
            value_name='Queries per second'
        )
        subset_melted['Type'] = subset_melted['Type'].map({
            'upload_actual_client_count': 'Lost',
            'download_actual_client_count': 'Received'
        })

        sns.barplot(
            x='tunnel_name',
            y='Queries per second',
            hue='Type',
            data=subset_melted,
            dodge=False,
            palette={'Lost': 'lightgrey', 'Received': '#66C266'},
            order=custom_order
        )
        if len(dns_resolver) == 0:
            # plt.yscale('symlog')
            pass
        else:
            plt.ylim(0, 1100)
        if i == len(test_types) - 1:
            plt.legend(loc='upper right')
        else:
            plt.legend().remove()
        plt.xticks(rotation=60, ha='right')
        if i == 1:
            plt.xlabel('Tunnel Name')
        else:
            plt.xlabel('')
        if i != 0:
            plt.ylabel('')

    plt.tight_layout()
    plt.show()



@cli.command()
def delete(
        task_id: Annotated[str, typer.Option(help="Task ID")] = None,
        db: Annotated[str, opt_type_db] = opt_type_db_default,
):
    connection = psycopg2.connect(db)

    if task_id:
        connection.cursor().execute("""
            DELETE FROM task_metadata
            WHERE
                task_id = %s"""
        , (task_id, ))

    # vacuum and re-analyze the table
    connection.commit()
    connection.autocommit = True
    connection.cursor().execute("VACUUM ANALYZE task_metadata")
    connection.cursor().execute("VACUUM ANALYZE packet_data")

@cli.command()
def run(
        test: Annotated[List[Test], opt_type_test],
        tunnel: Annotated[List[Tunnel], opt_type_tunnel],
        dns_resolver: Annotated[List[str], opt_type_dns_resolver] = [],
        dns_resolver_rate_limit: Annotated[int, opt_type_dns_resolver_rate_limit] = 0,
        net_em_client: Annotated[List[NetEmClient], opt_type_net_em_client] = [NetEmClient.university],
        net_em_server: Annotated[List[NetEmServer], opt_type_net_em_server] = [NetEmServer.international],
        website_path: Annotated[str, opt_type_website_path] = "",
        db: Annotated[str, opt_type_db] = opt_type_db_default,
        dry_run: Annotated[bool, opt_dry_run] = False,
):
    from .tasks.run import run

    if len(test) == 0:
        raise Exception("No test selected")

    for tunnel_t in tunnel:
        for net_em_client_t in net_em_client:
            for net_em_server_t in net_em_server:
                for test_t in test:
                    connection = psycopg2.connect(db)

                    logger.info(f"Running {test_t.value} test")
                    result = run.apply(
                        args=[connection, tunnel_t, net_em_client_t, net_em_server_t, dns_resolver, dns_resolver_rate_limit, opt_verbose, test_t, website_path, opt_client_docker_host,
                              opt_server_docker_host, dry_run],
                    )
                    logger.info(f"Task result: {result}")


@cli.command()
def destroy():
    safe_log(logger.info, f"Downing entire compose project")
    for host in {opt_client_docker_host, opt_server_docker_host}:
        downer = DockerClient(compose_files=["docker-compose-common.yaml"], host=host)
        log_stream_logs(downer.compose.down, timeout=1, remove_orphans=True)


@cli.command()
def build(
        tunnel: Annotated[List[Tunnel], opt_type_tunnel],
        dns_resolver: Annotated[List[str], opt_type_dns_resolver] = [],
):
    local_dns_resolvers_docker_controller = new_local_dns_resolvers_docker_controller(dns_resolver)
    if local_dns_resolvers_docker_controller:
        local_dns_resolvers_docker_controller.compose.build(services="dns-resolver")

    env = Env([], local=opt_client_docker_host == opt_server_docker_host, server_host=urlparse(
        opt_server_docker_host).hostname if opt_server_docker_host else opt_server_docker_host, test=Test.latency)

    for tunnel_t in tunnel:
        docker_controller = new_docker_controller(env, tunnel_t, client_host=opt_client_docker_host,
                                                  server_host=opt_server_docker_host)

        log_stream_logs(docker_controller.client.compose.build, progress="plain")
        log_stream_logs(docker_controller.server.compose.build, progress="plain")


@cli.callback()
def main(
        client_docker_host: Annotated[str, opt_type_docker_host] = None,
        server_docker_host: Annotated[str, opt_type_docker_host] = None,
        verbose: bool = False,
):
    global opt_client_docker_host, opt_server_docker_host, opt_verbose
    opt_client_docker_host = client_docker_host
    opt_server_docker_host = server_docker_host
    opt_verbose = verbose

