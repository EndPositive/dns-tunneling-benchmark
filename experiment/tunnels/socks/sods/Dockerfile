FROM gcc:14.2.0-bookworm as builder

WORKDIR /usr/src/app

COPY sods .

RUN make

FROM debian:bookworm-slim

# install socat without cache
RUN apt-get update && apt-get install -y socat

COPY --from=builder /usr/src/app/sods/sods /usr/local/bin/sods
COPY --from=builder /usr/src/app/sdt/sdt /usr/local/bin/sdt
COPY --from=builder /usr/src/app/ds/ds /usr/local/bin/ds
