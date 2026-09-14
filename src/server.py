"""Local web interface.

Built on the standard library's http.server — no Flask, FastAPI, Streamlit, or Gradio.
The dependency set for this project is deliberately small (see CLAUDE.md), and a GUI is
not a good reason to add a web framework when ~200 lines of stdlib does the job.

The server binds to localhost only. Models are loaded once, on the first question, and
reused for the life of the process.

    python -m src.server            # then open http://127.0.0.1:8000
    python -m src.server --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .config import REPO_ROOT, load_config


from .ingest import (
    IngestError,
    add_document,
    delete_document,
    extract_pdf_text,
    list_documents,
)
from .pipeline import Pipeline
from .retrieval import Retriever

def server_config():
    """Config for the reading room, which serves real uploaded papers.

    The file defaults were calibrated on a 24-document hand-built corpus. Two of them are
    actively wrong for real papers, and both were measured weeks before anyone noticed the
    interface was still using them:

      * title prefixing helps on short conceptual doc ids and *reverses* on real papers,
        where a title slug repeats on every chunk (0.176 -> 0.004);
      * the 0.15 support threshold sits above the median best-support of answerable
        questions on real papers (0.080), so the interface refused questions its own
        retrieved evidence covered.

    Overridden here rather than in the file so every archived experiment stays
    reproducible against the settings it recorded.
    """
    cfg = load_config()
    block = cfg.get_path("server") or {}

    cfg["evaluation"] = dict(
        cfg["evaluation"],
        contextualize=bool(block.get("contextualize", False)),
    )
    scorer = cfg.get_path("evaluation.scorer", "relevance")
    scorers = dict(cfg["evaluation"]["scorers"])
    scorers[scorer] = dict(
        scorers[scorer], threshold=float(block.get("support_threshold", 0.01))
    )
    cfg["evaluation"] = dict(cfg["evaluation"], scorers=scorers)

    if block.get("generation"):
        cfg["models"] = dict(cfg["models"], generation=block["generation"])

    cfg["abstain"] = dict(
        cfg["abstain"],
        gate=block.get("abstain_gate", "retrieval"),
        low_retrieval_score=float(block.get("retrieval_floor", 0.40)),
    )
    return cfg


logger = logging.getLogger(__name__)

WEB_DIR = REPO_ROOT / "web"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


class Engine:
    """Owns the pipeline and rebuilds the index when the paper set changes.

    Loading is lazy and guarded by a lock: the first question pays the model-loading
    cost, and concurrent requests wait rather than each loading their own copy.
    """

    def __init__(self) -> None:
        self.cfg = server_config()
        self._pipeline: Pipeline | None = None
        self._lock = threading.Lock()
        self.state = "idle"
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def pipeline(self) -> Pipeline:
        with self._lock:
            if self._pipeline is None:
                self.state = "loading"
                try:
                    self._pipeline = Pipeline(cfg=self.cfg)
                    self._pipeline.retriever.build()
                    self.state = "ready"
                    self.error = None
                except Exception as exc:  # noqa: BLE001 - reported to the browser
                    self.state = "error"
                    self.error = str(exc)
                    raise
            return self._pipeline

    def reindex(self) -> None:
        """Rebuild retrieval after papers are added or removed.

        Only the retriever is rebuilt; the scorer and generator weights are untouched,
        so adding a paper costs an embedding pass rather than a full reload.
        """
        with self._lock:
            if self._pipeline is None:
                return
            self.state = "loading"
            self._pipeline.retriever = Retriever(cfg=self.cfg)
            self._pipeline.retriever.build()
            self.state = "ready"


ENGINE = Engine()


def friendly_error(exc: Exception) -> str:
    """Turn an exception into something a reader can act on."""
    if isinstance(exc, IngestError):
        return str(exc)
    text = str(exc)
    if "no non-empty" in text or "not found" in text.lower():
        return "There are no papers loaded yet. Add one to get started."
    if "connect" in text.lower() or "resolve" in text.lower():
        return (
            "A model still needs downloading and the network is unavailable. "
            "Reconnect and try again."
        )
    return f"Something went wrong: {text}"


class Handler(BaseHTTPRequestHandler):
    server_version = "UncertaintyGuidedRAG"

    def log_message(self, fmt, *args):  # noqa: A003 - quieter default logging
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # --- helpers -----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise IngestError("That file is too large — 32 MB is the limit.")
        return self.rfile.read(length) if length else b""

    def _serve_file(self, name: str) -> None:
        path = WEB_DIR / name
        if not path.exists():
            self._send(404, b"Not found", "text/plain")
            return
        kind = "text/html; charset=utf-8" if path.suffix == ".html" else "text/plain"
        self._send(200, path.read_bytes(), kind)

    # --- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = urlparse(self.path).path
        try:
            if route in ("/", "/index.html"):
                self._serve_file("index.html")
            elif route == "/api/papers":
                self._json({"papers": list_documents(ENGINE.cfg)})
            elif route == "/api/status":
                self._json(
                    {
                        "state": ENGINE.state,
                        "ready": ENGINE.ready,
                        "error": ENGINE.error,
                        "papers": len(list_documents(ENGINE.cfg)),
                    }
                )
            else:
                self._send(404, b"Not found", "text/plain")
        except Exception as exc:  # noqa: BLE001
            logger.exception("GET %s failed", route)
            self._json({"error": friendly_error(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route == "/api/ask":
                self._handle_ask()
            elif route == "/api/papers":
                self._handle_add_paper()
            elif route == "/api/extract":
                self._handle_extract()
            elif route == "/api/papers/delete":
                self._handle_delete_paper()
            else:
                self._send(404, b"Not found", "text/plain")
        except IngestError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("POST %s failed", route)
            self._json({"error": friendly_error(exc)}, 500)

    def _handle_ask(self) -> None:
        payload = json.loads(self._read_body() or b"{}")
        query = (payload.get("query") or "").strip()
        if not query:
            self._json({"error": "Type a question first."}, 400)
            return

        if not list_documents(ENGINE.cfg):
            self._json({"error": "Add a paper before asking a question."}, 400)
            return

        mode = payload.get("mode") or "uncertainty_guided"
        answer, trace = ENGINE.pipeline().run(query, compression_mode=mode)

        self._json(
            {
                "query": query,
                "answer": answer.text,
                "abstained": answer.abstained,
                "branch": answer.branch,
                "unsupported": answer.unsupported_claims,
                "trace": {
                    "baseline_tokens": trace.baseline_tokens,
                    "final_tokens": trace.final_tokens,
                    "reduction": round(trace.reduction, 4),
                    "restoration_triggered": trace.restoration_triggered,
                    "token_counts_exact": trace.token_counts_exact,
                    "claims": [asdict(c) for c in trace.claims],
                },
            }
        )

    def _handle_extract(self) -> None:
        """Pull text out of a PDF without storing it.

        The upload flow used to add the paper the moment a file was chosen, while the
        title and text fields stayed on screen — so the natural next click on "Add paper"
        submitted an empty form and produced "That is too short to answer questions from"
        on a paper that had in fact been added. Extraction and commitment are now separate:
        choosing a file fills the form, and "Add paper" is the only thing that saves.
        """
        body = self._read_body()
        text = extract_pdf_text(body)
        self._json({"text": text, "characters": len(text)})

    def _handle_add_paper(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        body = self._read_body()

        if "application/pdf" in content_type:
            title = self.headers.get("X-Paper-Title") or "Untitled paper"
            text = extract_pdf_text(body)
        else:
            payload = json.loads(body or b"{}")
            title = payload.get("title") or ""
            text = payload.get("text") or ""

        doc_id = add_document(title, text, cfg=ENGINE.cfg)
        ENGINE.reindex()
        self._json({"doc_id": doc_id, "papers": list_documents(ENGINE.cfg)})

    def _handle_delete_paper(self) -> None:
        payload = json.loads(self._read_body() or b"{}")
        removed = delete_document(payload.get("doc_id") or "", cfg=ENGINE.cfg)
        if removed:
            ENGINE.reindex()
        self._json({"removed": removed, "papers": list_documents(ENGINE.cfg)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local web interface.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Reading room running at {url}")
    print(f"{len(list_documents(ENGINE.cfg))} papers loaded. Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
