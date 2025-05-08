#!/usr/bin/env python3

import csv
import getopt
import socket
import sys
import time

TIME_LIMIT = 20 * 60
SOCK_TIMEOUT = 60
MILESTONE_INTERVAL = 1024 * 1024


def receive(sock, writerow):
    writerow("start", None, 0)
    deadline = time.monotonic() + TIME_LIMIT
    sum = 0
    while True:
        now = time.monotonic()
        if now >= deadline:
            writerow("timeout", "Time limit of 10 minutes reached", 0)
            break
            
        sock.settimeout(SOCK_TIMEOUT)
        try:
            data = sock.recv(1024 * 1024)
            writerow("read", "", len(data))
            sum += len(data)
        except Exception as read_err:
            writerow("read", str(read_err), 0)
            break
        if not data:
            writerow("eof", "", 0)
            break


def main():
    _, args = getopt.gnu_getopt(sys.argv[1:], "")

    mode, interface_name, address = args

    csv_writer = csv.writer(sys.stdout)

    start_time = 0
    def writerow(event, error, amount):
        elapsed_ms = int((time.time_ns() - start_time))
        csv_writer.writerow([elapsed_ms, event, error or "", str(amount)])
        sys.stdout.flush()

    host, port = address.rsplit(":", 1)
    port = int(port)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if mode == "client":
        if not interface_name == "none":
            s.setsockopt(socket.SOL_SOCKET, 25, interface_name.encode())
        s.connect((host, port))
        start_time = time.time_ns()
        receive(s, writerow)
    elif mode == "server":
        s.bind((host, port))
        s.listen(1)
        writerow("listen", None, 0)
        client_sock, client_addr = s.accept()
        start_time = time.time_ns()
        writerow("accept", f"{client_addr[0]}:{client_addr[1]}", 0)
        receive(client_sock, writerow)
        client_sock.close()
    
    s.close()


if __name__ == "__main__":
    main()
