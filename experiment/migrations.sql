CREATE TYPE relevance_enum AS ENUM (
    'Perceived Upload (server)',
    'Perceived Upload (client)',
    'Actual Upload (server)',
    'Actual Upload (client)',
    'Perceived Download (server)',
    'Perceived Download (client)',
    'Actual Download (server)',
    'Actual Download (client)'
);

CREATE TABLE IF NOT EXISTS task_metadata
(
    task_id          varchar(155) PRIMARY KEY,
    tunnel_name      TEXT,
    dns_resolvers    TEXT[],
    test_type        TEXT,
    netem_name       TEXT,
    netem            JSONB
);

CREATE TABLE IF NOT EXISTS packet_data
(
    id                 SERIAL PRIMARY KEY,
    task_id            varchar(155),
    client             BOOLEAN,
    frame_time         DOUBLE PRECISION,
    frame_time_floor   INTEGER GENERATED ALWAYS AS (floor(frame_time)) STORED,
    frame_len          INTEGER,
    frame_interface_id INTEGER,
    ip_src             TEXT,
    ip_dst             TEXT,
    tcp_srcport        INTEGER,
    tcp_dstport        INTEGER,
    udp_srcport        INTEGER,
    udp_dstport        INTEGER,
    ip_proto           TEXT,
    tcp_flags          TEXT,
    relevance          relevance_enum GENERATED ALWAYS AS (
        CASE
            WHEN (tcp_dstport = 5201)
                AND NOT client
                THEN 'Perceived Upload (server)'::relevance_enum
            WHEN (tcp_dstport = 5201)
                AND client
                THEN 'Perceived Upload (client)'::relevance_enum
            WHEN (udp_dstport = 53)
                AND NOT client
                THEN 'Actual Upload (server)'::relevance_enum
            WHEN (udp_dstport = 53)
                AND client
                THEN 'Actual Upload (client)'::relevance_enum
            WHEN (tcp_srcport = 5201)
                AND not client
                THEN 'Perceived Download (server)'::relevance_enum
            WHEN (tcp_srcport = 5201)
                AND client
                THEN 'Perceived Download (client)'::relevance_enum
            WHEN (udp_srcport = 53)
                AND NOT client
                THEN 'Actual Download (server)'::relevance_enum
            WHEN (udp_srcport = 53)
                AND client
                THEN 'Actual Download (client)'::relevance_enum
            END ) STORED,
    FOREIGN KEY (task_id) REFERENCES task_metadata (task_id) ON DELETE CASCADE
);

CREATE OR REPLACE VIEW full_packet_data AS
SELECT pd.id,
       pd.task_id,
       pd.client,
       tm.tunnel_name,
       tm.dns_resolvers,
       tm.test_type,
       tm.netem_name,
       tm.netem,
       pd.frame_time_floor,
       pd.frame_len,
       pd.ip_src,
       pd.ip_dst,
       pd.tcp_srcport,
       pd.tcp_dstport,
       pd.udp_srcport,
       pd.udp_dstport,
       pd.ip_proto,
       pd.tcp_flags,
       pd.relevance
FROM packet_data pd
         JOIN
     task_metadata tm
     ON
         pd.task_id = tm.task_id;


CREATE INDEX idx_packet_data_relevance
    ON packet_data (relevance);

CREATE INDEX idx_packet_data_task_id
    ON packet_data (task_id);

CREATE INDEX idx_packet_data_composite
    ON packet_data (task_id, relevance, frame_time_floor);

CREATE INDEX idx_task_metadata_composite
    ON task_metadata (tunnel_name, test_type, netem_name);

ANALYZE;
