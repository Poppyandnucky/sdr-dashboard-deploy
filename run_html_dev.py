"""Start the primitive HTML frontend and Streamlit backend together."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


class ComponentRequestHandler(SimpleHTTPRequestHandler):
    """Serve component assets with headers required by Streamlit dev mode."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()


def serve_frontend(port: int) -> ThreadingHTTPServer:
    handler = partial(ComponentRequestHandler, directory=str(FRONTEND))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-port", type=int, default=3000)
    parser.add_argument("--streamlit-port", type=int, default=8501)
    args = parser.parse_args()

    server = serve_frontend(args.frontend_port)
    environment = os.environ.copy()
    environment["SDR_USE_HTML_UI"] = "1"
    environment["SDR_HTML_UI_URL"] = f"http://127.0.0.1:{args.frontend_port}"

    bundled_parameters = ROOT / "data" / "SDR Parameters.xlsx"
    if "SDR_PARAMS_PATH" not in environment and bundled_parameters.exists():
        environment["SDR_PARAMS_PATH"] = str(bundled_parameters)

    command = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "SDR_Dash_TI.py"),
        "--server.port", str(args.streamlit_port),
    ]

    print(f"Frontend: http://127.0.0.1:{args.frontend_port}")
    print(f"Application: http://localhost:{args.streamlit_port}")
    try:
        return subprocess.call(command, cwd=ROOT, env=environment)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
