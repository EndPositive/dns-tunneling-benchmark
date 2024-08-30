FROM perl:5.40.0-slim-threaded-buster

WORKDIR /usr/src/app

RUN apt update && apt upgrade -y

RUN apt install -y libwww-perl libclone-perl libnet-dns-perl libmime-base32-perl libdigest-crc-perl socat

ENV PERL5LIB=/usr/share/perl5/

COPY . .
