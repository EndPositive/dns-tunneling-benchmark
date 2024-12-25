FROM debian:buster-slim as downloader

WORKDIR /usr/src/app

RUN apt update && apt install -y unzip wget && \
    wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/dnscapy/dnscapy-0-99b.zip && \
    unzip dnscapy-0-99b.zip

FROM python:2.7.18-slim-buster

WORKDIR /usr/src/app

RUN apt update && apt install tcpdump socat -y

RUN pip install scapy==2.3.1

COPY --from=downloader /usr/src/app/dnscapy_client.py /usr/src/app
COPY --from=downloader /usr/src/app/dnscapy_server.py /usr/src/app

