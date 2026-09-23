"""Campus-only TCP entry to the running local backend; no backend restart needed."""
import argparse
import select
import signal
import socket
import socketserver

try:
    from .access_policy import is_campus_peer
except ImportError:
    from access_policy import is_campus_peer


class CampusProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        client = self.request
        client.settimeout(30)
        if not is_campus_peer(self.client_address[0]):
            body = b"Campus network access only.\n"
            try:
                client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\nContent-Length: "
                               + str(len(body)).encode() + b"\r\n\r\n" + body)
                client.shutdown(socket.SHUT_WR)
                client.recv(4096)
            except OSError:
                pass
            return
        try:
            with socket.create_connection(("127.0.0.1", self.server.target_port), timeout=10) as upstream:
                upstream.settimeout(30)
                peers = {client: upstream, upstream: client}
                readers = list(peers)
                while readers:
                    ready, _, _ = select.select(readers, [], [], 120)
                    if not ready:
                        return
                    for source in ready:
                        data = source.recv(65536)
                        if data:
                            peers[source].sendall(data)
                        else:
                            readers.remove(source)
                            peers[source].shutdown(socket.SHUT_WR)
        except OSError:
            return


class CampusProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address, target_port=8765):
        self.target_port = target_port
        super().__init__(address, CampusProxyHandler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Capture host's campus IPv4 address")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--target-port", type=int, default=8765)
    args = parser.parse_args()
    if not is_campus_peer(args.host):
        parser.error("--host must be a 10.0.0.0/8 campus address")
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    with CampusProxyServer((args.host, args.port), args.target_port) as server:
        print(f"Campus entry {args.host}:{args.port} -> 127.0.0.1:{args.target_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
