FROM python:3.12-alpine

WORKDIR /usr/src/app

RUN dd if=/dev/zero of=1b.txt bs=1 count=1
RUN dd if=/dev/zero of=1kb.txt bs=1024 count=1
RUN dd if=/dev/zero of=1mb.txt bs=1048576 count=1
RUN dd if=/dev/zero of=10mb.txt bs=10485760 count=1
RUN dd if=/dev/zero of=100mb.txt bs=104857600 count=1

EXPOSE 8000

CMD ["python", "-m", "http.server", "8000"]
