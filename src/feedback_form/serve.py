#!/usr/bin/env python3
"""Pkg 3: HTTP server for the plain-English translation review tool
(feedback form) [last updated 2026-10-23]

This file owns no review logic and no markup. It serves the page
dynamically — review_template.html with empty data slots, which then fetches
its data from the JSON API below — so metric or dataset changes are live
on the next browser refresh, with no build step. Everything is read from
disk per request.

  GET  /, /index.html, /review   the page (raw template, boots from the API)
  GET  /healthz                  liveness check for hosting platforms
  GET  /api/axes                 review axes (active metrics of the shared
                                 settings + labels/groups, via build.build_axes)
  GET  /api/config               which prompt's draft is reviewed + prompt
                                 labels (settings selected_prompts / prompts.json,
                                 via build.build_form_config)
  GET  /api/records?n=&seed=     a random sample of review_data.jsonl:
                                 {"total", "sample_size", "seed", "records"}.
                                 n defaults to settings feedback_form.sample_size
                                 and is capped at max_sample_size; n=0 means all
                                 (only if the file is within the cap). The same
                                 (n, seed) returns the same records, so a
                                 reviewer's set is stable across reloads. The
                                 file is indexed, not loaded (records.py), so
                                 it may hold the full UHRI export.
  POST /api/reviews              store a submitted review export as
                                 src/feedback_form/exports/review-<utc>-<rand>.json
                                 (payload verbatim, same shape as the page's
                                 manual JSON export; exports/ is gitignored)

The standalone baked page (build.py -> translation_review.html) still exists
for no-server distribution (email / static host); it ignores the API and
keeps the manual copy/download export only.

Run (from the repo root):
    python -m src.feedback_form.serve                   # host/port from the settings
    python -m src.feedback_form.serve --port 9000 --host 0.0.0.0

Environment (for containers / Cloud Run, see the Dockerfile and README):
    PORT                  overrides the port from the settings (Cloud Run sets it)
    FEEDBACK_EXPORTS_DIR  where submissions are written (default: exports/
                          next to this file; point it at a mounted bucket/volume
                          so submissions survive a redeploy)
    FEEDBACK_RECORDS_PATH the review_data.jsonl to serve (default: the one next
                          to this file, baked into the image; point it at a
                          mounted bucket for datasets too big to bake)

Open Issues:
    - Submitted exports are trusted JSON from reviewers on the network the
      server listens on; there is no auth in the app itself. Fine on a
      private network or behind an authenticating proxy (e.g. Cloud Run +
      Identity-Aware Proxy), not for the open internet as-is.

Resumption:
    New route: one more branch in Handler.do_GET/do_POST.
    Per-reviewer assignment (no overlap between reviewers) would replace the
    seed with a server-side cursor; records.py already has the offset index.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

from ..settings import load_settings
from .build import build_axes, build_form_config
from .records import RecordStore

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "review_template.html"
RECORDS = Path(os.environ.get("FEEDBACK_RECORDS_PATH") or HERE / "review_data.jsonl")
EXPORTS_DIR = Path(os.environ.get("FEEDBACK_EXPORTS_DIR") or HERE / "exports")
STORE = RecordStore(RECORDS)

MAX_SUBMIT_BYTES = 8 * 1024 * 1024   # generous: full-dataset review ~ a few MB


def _int_param(query: dict, key: str) -> Optional[int]:
    """First value of a query parameter as an int, or None if absent/invalid."""
    try:
        return int(query[key][0])
    except (KeyError, IndexError, ValueError):
        return None


def sample_records(store: RecordStore, n: Optional[int], seed: Optional[int]) -> dict:
    """The /api/records payload: a seeded random sample within the settings' bounds.

    Args:
        store: Where the records come from.
        n: Requested sample size; None -> settings sample_size; 0 -> all,
            which is capped at max_sample_size like any other request.
        seed: Reproducibility seed; None -> a fresh random one, echoed back
            so the page can keep it.

    Returns:
        {"total", "sample_size", "seed", "records"}.
    """
    config = load_settings()["feedback_form"]
    total = store.count()
    cap = max(1, int(config.get("max_sample_size", 500)))
    if n is None:
        n = int(config.get("sample_size", 50))
    if n <= 0 or n > total:
        n = total
    n = min(n, cap)
    if seed is None:
        seed = secrets.randbelow(2**32)
    records = store.sample(n, seed) if n < total else store.sample(0)
    return {"total": total, "sample_size": len(records), "seed": seed, "records": records}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        print("[http] " + (fmt % args), flush=True)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        path = url.path

        if path in ("/", "/index.html", "/review"):
            if not TEMPLATE.is_file():
                self._send(503, b"review_template.html missing", "text/plain; charset=utf-8")
                return
            self._send(200, TEMPLATE.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/healthz":
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return

        if path == "/api/axes":
            self._json(200, build_axes())
            return

        if path == "/api/config":
            self._json(200, build_form_config())
            return

        if path == "/api/records":
            if not STORE.exists():
                self._json(503, {"error": f"{RECORDS.name} not generated yet - "
                                          "run: python -m src.feedback_form.make_data"})
                return
            query = parse_qs(url.query)
            self._json(200, sample_records(STORE, _int_param(query, "n"), _int_param(query, "seed")))
            return

        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/reviews":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_SUBMIT_BYTES:
            self._json(413 if length else 400, {"error": "missing or oversized body"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self._json(400, {"error": f"invalid JSON: {exc}"})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            self._json(400, {"error": "expected the page's JSON export ({records: [...]})"})
            return

        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"review-{stamp}-{secrets.token_hex(3)}.json"
        (EXPORTS_DIR / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[reviews] stored {name} ({len(payload['records'])} records)", flush=True)
        self._json(201, {"ok": True, "stored": name})


def main() -> None:
    config = load_settings()["feedback_form"]
    parser = argparse.ArgumentParser(description="Serve the translation review tool.")
    parser.add_argument("--host", default=config["host"])
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT") or config["port"]))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving review tool on http://{args.host}:{args.port}  "
          f"(submissions -> {EXPORTS_DIR})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
