import http.server
import socketserver
import os

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    # SPA routing - redirect 404 to index.html
    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.exists(path):
            self.path = '/index.html'
        return super().do_GET()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving UI at http://localhost:{PORT}")
    httpd.serve_forever()
