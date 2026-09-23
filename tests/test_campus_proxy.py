import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_backend.campus_proxy import CampusProxyServer


class PeerProxy(CampusProxyServer):
    peer = '10.230.194.204'

    def get_request(self):
        connection, address = super().get_request()
        return connection, (self.peer, address[1])


class Echo(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        type(self).calls += 1
        data = self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


class CampusProxyTest(unittest.TestCase):
    @patch.dict('os.environ', {'ORBBEC_OPERATOR_NAT_NETWORKS': '10.192.35.102/32'})
    def test_forward_body_and_block_non_campus_without_contacting_backend(self):
        backend = ThreadingHTTPServer(('127.0.0.1', 0), Echo)
        proxy = PeerProxy(('127.0.0.1', 0), backend.server_port)
        servers = [backend, proxy]
        threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
        for thread in threads:
            thread.start()
        try:
            data = bytes(range(256)) * 8192
            url = f'http://127.0.0.1:{proxy.server_address[1]}/upload'
            with urlopen(Request(url, data=data), timeout=5) as response:
                self.assertEqual(response.read(), data)
            calls = Echo.calls
            for peer in ('192.168.50.177', '10.192.35.102'):
                proxy.peer = peer
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url, data=b'denied', headers={'Host': '10.1.2.3', 'X-Forwarded-For': '10.1.2.3'}), timeout=5)
                self.assertEqual(error.exception.code, 403)
                self.assertEqual(Echo.calls, calls)
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(3)


if __name__ == '__main__':
    unittest.main()
