#!/usr/bin/env python3
"""Thin HTTP server for the plain-English translation review tool.

Same transport-only pattern as web_demo.py on the retrieval-eval-prototype
branch: this file owns no review logic and no markup. The page itself
(translation_review.html) is fully self-contained — data and review axes are
baked in by build_review_tool.py — so the server only hands out that file.

Run:
    python review/serve_review.py                   # http://127.0.0.1:8010
    python review/serve_review.py --port 9000 --host 0.0.0.0

Rebuild the page first if the data or dimensions changed:
    python review/make_review_data.py && python review/build_review_tool.py

Open Issues:
    - Reviewer answers still live in the reviewer's browser and come back via
      the page's export button; the server does not collect them. A POST
      endpoint that receives exports would save the email round-trip.

Resumption:
    When this merges with the retrieval demo, fold do_GET into web_demo.py's
    Handler as a "/review" route and delete this file.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent          # code/review
PAGE = HERE / "translation_review.html"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        print("[http] " + (fmt % args), flush=True)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] not in ("/", "/index.html", "/review"):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        if not PAGE.is_file():
            self._send(
                503,
                b"translation_review.html not built yet - run: "
                b"python review/build_review_tool.py",
                "text/plain; charset=utf-8",
            )
            return
        self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the translation review page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving review tool on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
