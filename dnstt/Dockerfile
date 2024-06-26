FROM golang:1.22.4-bookworm as builder

WORKDIR /usr/src/app

RUN --mount=type=cache,target=/go/pkg/mod/ go install www.bamsoftware.com/git/dnstt.git/dnstt-client@v1.20240513.0
RUN --mount=type=cache,target=/go/pkg/mod/ go install www.bamsoftware.com/git/dnstt.git/dnstt-server@v1.20240513.0

RUN dnstt-server -gen-key -privkey-file server.key -pubkey-file server.pub

FROM debian:bookworm-slim

WORKDIR /usr/src/app

RUN apt update && apt upgrade -y

COPY --from=builder /go/bin/dnstt-client /usr/local/bin/dnstt-client
COPY --from=builder /go/bin/dnstt-server /usr/local/bin/dnstt-server
COPY --from=builder /usr/src/app/server.key server.key
COPY --from=builder /usr/src/app/server.pub server.pub
