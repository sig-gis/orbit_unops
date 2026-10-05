import http.server
import socketserver
import os

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if load_dotenv is not None:
    load_dotenv(os.path.join(ROOT_DIR, ".env"))

PORT = int(os.getenv("PORT", "3000"))
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
API_BASE_URL = os.getenv("ORBIT_API_BASE_URL", "http://localhost:8000").rstrip("/")
GCS_BUCKET = os.getenv("GCS_BUCKET", "example-bucket")
NLC_DEMO_ASSET_ID = os.getenv(
    "NLC_DEMO_ASSET_ID",
    "projects/example-project-id/assets/example-asset",
)
NLC_DEMO_COG_URL = os.getenv(
    "NLC_DEMO_COG_URL",
    "https://storage.googleapis.com/example-bucket/example/path/example.tif",
)
GCP_PROJECT = os.getenv("GCP_PROJECT", "example-project-id")
NLC_CLOUD_PROJECT = GCP_PROJECT
NLC_ASSET_ROOT = os.getenv(
    "NLC_ASSET_ROOT",
    f"projects/{NLC_CLOUD_PROJECT}/assets/space_for_time_tasking",
)
NLC_RESULTS_PREFIX = os.getenv("NLC_RESULTS_PREFIX", "space_for_time_tasking/results")

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
                f"window.ORBIT_CONFIG.NLC_DEMO_ASSET_ID = {NLC_DEMO_ASSET_ID!r};\n"
                f"window.ORBIT_CONFIG.NLC_DEMO_COG_URL = {NLC_DEMO_COG_URL!r};\n"
                f"window.ORBIT_CONFIG.GCP_PROJECT = {GCP_PROJECT!r};\n"
                f"window.ORBIT_CONFIG.NLC_CLOUD_PROJECT = {NLC_CLOUD_PROJECT!r};\n"
                f"window.ORBIT_CONFIG.NLC_ASSET_ROOT = {NLC_ASSET_ROOT!r};\n"
                f"window.ORBIT_CONFIG.NLC_RESULTS_PREFIX = {NLC_RESULTS_PREFIX!r};\n"
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
