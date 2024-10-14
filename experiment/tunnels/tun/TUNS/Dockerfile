FROM ruby:3.3.3-slim-bookworm

WORKDIR /usr/src/app

RUN apt update && apt upgrade -y

RUN apt install -y iptables net-tools

# install base32 ruby package
RUN gem install base32

COPY tuns .

ENV RUBYLIB=/usr/src/app/lib
