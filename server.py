"""Local development server to preview the Vercel UI and /api/simulate locally."""

import http.server
import os
import socketserver
from api.simulate import handler as SimulateHandler

PORT = 8000
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")


class LocalHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

    def do_POST(self):
        if self.path == "/api/simulate":
            SimulateHandler.do_POST(self)
        else:
            self.send_error(404)


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), LocalHandler) as httpd:
        print(f"Serving UI at http://localhost:{PORT}")
        httpd.serve_forever()
