FROM golang:1.22.4-bookworm AS builder

WORKDIR /usr/src/app

RUN --mount=type=cache,target=/go/pkg/mod/ go install www.bamsoftware.com/git/dnstt.git/dnstt-client@v1.20240513.0
RUN --mount=type=cache,target=/go/pkg/mod/ go install www.bamsoftware.com/git/dnstt.git/dnstt-server@v1.20240513.0


FROM debian:bookworm-slim

WORKDIR /usr/src/app

RUN apt update && apt upgrade -y

COPY --from=builder /go/bin/dnstt-client /usr/local/bin/dnstt-client
COPY --from=builder /go/bin/dnstt-server /usr/local/bin/dnstt-server

COPY server.key server.key
COPY server.pub server.pub
