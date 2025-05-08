CREATE TABLE IF NOT EXISTS task_metadata
(
    task_id                 varchar(155) PRIMARY KEY,
    tunnel_name             TEXT,
    dns_resolvers           JSONB,
    dns_resolver_rate_limit INTEGER,
    test_type               TEXT,
    netem_client_name       TEXT,
    netem_client            JSONB,
    netem_server_name       TEXT,
    netem_server            JSONB,
    website_path            TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS file_download_result
(
    id               SERIAL PRIMARY KEY,
    task_id          varchar(155) REFERENCES task_metadata (task_id) ON DELETE CASCADE,
    time BIGINT,
    event_type       TEXT,
    error            TEXT,
    bytes_received   BIGINT
);

CREATE TABLE IF NOT EXISTS unbound_metrics
(
    id          SERIAL PRIMARY KEY,
    task_id     varchar(155) REFERENCES task_metadata (task_id) ON DELETE CASCADE,
    time        BIGINT,
    resolver_id BIGINT,
    metric      TEXT,
    value       BIGINT
);

CREATE TABLE IF NOT EXISTS browsing_test_results
(
    id      SERIAL PRIMARY KEY,
    task_id varchar(155) REFERENCES task_metadata (task_id) ON DELETE CASCADE,
    avg     INT,
    min     INT,
    med     INT,
    max     INT,
    p90     INT,
    p95     INT
);

CREATE TABLE IF NOT EXISTS speedtest_result
(
    id      SERIAL PRIMARY KEY,
    task_id varchar(155) REFERENCES task_metadata (task_id) ON DELETE CASCADE,
    latency FLOAT,
    jitter  FLOAT
);

CREATE OR REPLACE VIEW full_speedtest_result AS
SELECT task_metadata.*,
       speedtest_result.latency,
       speedtest_result.jitter
FROM speedtest_result
         INNER JOIN
     task_metadata ON speedtest_result.task_id = task_metadata.task_id;

CREATE TABLE IF NOT EXISTS packet_data
(
    id                 SERIAL PRIMARY KEY,
    task_id            varchar(155) REFERENCES task_metadata (task_id) ON DELETE CASCADE,
    client             BOOLEAN,
    frame_time         DOUBLE PRECISION,
    frame_time_floor   INTEGER GENERATED ALWAYS AS (floor(frame_time)) STORED,
    tcp_len            INTEGER,
    tcp_srcport        INTEGER,
    tcp_dstport        INTEGER,
    udp_length         INTEGER,
    udp_srcport        INTEGER,
    udp_dstport        INTEGER,
    len_upload_perceived_client INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN (tcp_dstport = 5201)
                AND client
                THEN tcp_len
            ELSE 0
            END ) STORED,
    len_upload_perceived_server INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN (tcp_dstport = 5201)
                AND NOT client
                THEN tcp_len
            ELSE 0
            END ) STORED,
    len_upload_actual_client INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN (udp_dstport = 53)
                AND client
                THEN udp_length
            ELSE 0
            END ) STORED,
    len_upload_actual_server INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN (udp_dstport = 53)
                AND NOT client
                THEN udp_length
            ELSE 0
            END ) STORED,
    len_download_perceived_client INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN (tcp_srcport = 5201)
                AND client
                THEN tcp_len
            ELSE 0
            END ) STORED,
    len_download_perceived_server INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN (tcp_srcport = 5201)
                AND NOT client
                THEN tcp_len
            ELSE 0
            END ) STORED,
    len_download_actual_client INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN (udp_srcport = 53)
                AND client
                THEN udp_length
            ELSE 0
            END ) STORED,
    len_download_actual_server INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN (udp_srcport = 53)
                AND NOT client
                THEN udp_length
            ELSE 0
            END ) STORED
);

CREATE OR REPLACE VIEW full_packet_data AS
SELECT pd.id,
       pd.task_id,
       pd.client,
       tm.tunnel_name,
       tm.dns_resolvers,
       tm.test_type,
       tm.netem_client_name,
       tm.netem_client,
       tm.netem_server_name,
       tm.netem_server,
       pd.frame_time_floor,
       pd.tcp_len,
       pd.tcp_srcport,
       pd.tcp_dstport,
       pd.udp_length,
       pd.udp_srcport,
       pd.udp_dstport,
       pd.len_upload_perceived_client,
       pd.len_upload_perceived_server,
       pd.len_upload_actual_client,
       pd.len_upload_actual_server,
       pd.len_download_perceived_client,
       pd.len_download_perceived_server,
       pd.len_download_actual_client,
       pd.len_download_actual_server
FROM packet_data pd
         JOIN
     task_metadata tm
     ON
         pd.task_id = tm.task_id;

CREATE INDEX idx_packet_data_task_id
    ON packet_data (task_id);

CREATE INDEX idx_packet_data_composite
    ON packet_data (task_id, frame_time_floor);

CREATE INDEX idx_task_metadata_composite
    ON task_metadata (tunnel_name, test_type, netem_client_name, netem_server_name);

ANALYZE;
