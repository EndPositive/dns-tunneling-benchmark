FROM debian:bookworm-slim

WORKDIR /usr/src/app

RUN apt update && apt upgrade -y

RUN apt install -y dns2tcp socat gettext-base

COPY .dns2tcpdrc .dns2tcpdrc
