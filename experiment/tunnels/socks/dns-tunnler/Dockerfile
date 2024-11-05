FROM node:23.1.0-alpine3.19

WORKDIR /usr/src/app

RUN apk add --no-cache socat

RUN --mount=type=cache,target=/root/.npm \
    npm install --loglevel verbose native-dns@0.4.1 stdio request

COPY dns-tunnler .

ENTRYPOINT []
CMD []