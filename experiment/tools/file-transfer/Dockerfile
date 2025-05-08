FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

WORKDIR /usr/src/app

RUN apt update && apt install -y ncat socat

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt

COPY . .
