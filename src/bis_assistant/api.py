"""Stdlib-only HTTP API: POST /chat {query, lang?, force?, new_topic?, thread_id?} | GET /health.

Run: python -m bis_assistant.api

Honors the UI contract: mint/honor ``thread_id``, persist history, require
``X-Owner-Token`` on follow-ups. Legacy client-held ``context`` is accepted
as a one-turn migration bridge when no ``thread_id`` is sent.
"""
from __future__ import annotations
import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer

from .assistant import answer
from .i18n_privacy import redact
from . import threads as threadmod

PORT = 8000
MAX_BODY = 16 * 1024  # 16 KiB
_THREADS: dict[str, dict] = {}


class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, X-Owner-Token")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._json({})

    def do_GET(self):
        if self.path == "/health":
            return self._json({"ok": True})
        return self._json({"error": "use POST /chat"}, 404)

    def do_POST(self):
        if self.path != "/chat":
            return self._json({"error": "use POST /chat"}, 404)
        raw_len = self.headers.get("Content-Length", "0")
        try:
            n = int(raw_len)
        except (TypeError, ValueError):
            return self._json({"error": "invalid Content-Length"}, 400)
        if n < 0:
            return self._json({"error": "invalid Content-Length"}, 400)
        if n > MAX_BODY:
            leftover = n
            while leftover > 0:
                chunk = self.rfile.read(min(leftover, MAX_BODY))
                if not chunk:
                    break
                leftover -= len(chunk)
            return self._json({"error": "payload too large"}, 400)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json({"error": "invalid JSON"}, 400)
        if not isinstance(payload, dict):
            return self._json({"error": "invalid JSON"}, 400)
        q = (payload.get("query") or "").strip()
        if not q:
            return self._json({"error": "empty query"}, 400)
        if len(q) > 4000:
            return self._json({"error": "query too long"}, 400)
        owner = self.headers.get("X-Owner-Token") or ""
        tid = None if payload.get("new_topic") else payload.get("thread_id")
        history, rounds = [], 0
        if tid:
            rec = _THREADS.get(str(tid))
            if rec is None:
                return self._json({"error": "unknown thread; start a new topic",
                                   "code": "thread_gone"}, 410)
            if not hmac.compare_digest(rec["token"], owner):
                return self._json({"error": "owner token required",
                                   "code": "forbidden"}, 403)
            history, rounds = list(rec["history"]), int(rec["rounds"])
        elif isinstance(payload.get("context"), dict) and not payload.get("new_topic"):
            ctx_in = threadmod.normalize_context(payload["context"])
            history, rounds = ctx_in["history"], ctx_in["rounds"]
        ctx = {"history": history, "rounds": rounds,
               "force": bool(payload.get("force"))}
        resp = answer(q, payload.get("lang"), ctx)
        new_history = threadmod.normalize_context(resp.get("context"))["history"] \
            or threadmod.push_history(history, redact(q)[:2000])
        new_rounds = threadmod.rounds_from(resp.get("context"), default=rounds)
        if tid is None:
            tid = secrets.token_hex(8)
            token = secrets.token_urlsafe(32)
            _THREADS[tid] = {"history": new_history, "rounds": new_rounds,
                             "token": token}
            resp["owner_token"] = token
        else:
            rec = _THREADS[str(tid)]
            rec["history"] = new_history
            rec["rounds"] = new_rounds
        resp["thread_id"] = tid
        return self._json(resp)

    def log_message(self, *a):
        pass


def main():
    print(f"BIS Assistant API on http://127.0.0.1:{PORT}  (POST /chat)")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
