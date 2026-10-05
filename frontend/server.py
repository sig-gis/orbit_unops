import http.server
import socketserver
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PORT = int(os.getenv("PORT", "3000"))
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
API_BASE_URL = os.getenv("ORBIT_API_BASE_URL", "http://localhost:8000").rstrip("/")
GCS_BUCKET = os.environ["GCS_BUCKET"]
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    # SPA routing - redirect 404 to index.html
    def do_GET(self):
        if self.path == "/config.js":
            body = (
                "window.ORBIT_CONFIG = window.ORBIT_CONFIG || {};\n"
                f"window.ORBIT_CONFIG.API_BASE_URL = {API_BASE_URL!r};\n"
                f"window.ORBIT_CONFIG.GCS_BUCKET = {GCS_BUCKET!r};\n"
                f"window.ORBIT_CONFIG.GCP_PROJECT_ID = {GCP_PROJECT_ID!r};\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        path = self.translate_path(self.path)
        if not os.path.exists(path):
            self.path = '/index.html'
        return super().do_GET()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving UI at http://localhost:{PORT}")
    httpd.serve_forever()
