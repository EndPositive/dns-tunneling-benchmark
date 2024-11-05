FROM golang:1.23.2-bookworm AS builder

WORKDIR /usr/src/app

COPY dnstt .

RUN --mount=type=cache,target=/go/pkg/mod/ go build -o dnstt-client www.bamsoftware.com/git/dnstt.git/dnstt-client
RUN --mount=type=cache,target=/go/pkg/mod/ go build -o dnstt-server www.bamsoftware.com/git/dnstt.git/dnstt-server


FROM debian:bookworm-slim

WORKDIR /usr/src/app

RUN apt update && apt upgrade -y

COPY --from=builder /usr/src/app/dnstt-client/dnstt-client /usr/local/bin/dnstt-client
COPY --from=builder /usr/src/app/dnstt-server/dnstt-server /usr/local/bin/dnstt-server
