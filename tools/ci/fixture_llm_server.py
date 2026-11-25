#!/usr/bin/env python3
"""Deterministic local HTTP LLM fixture server for Chemistry tests, docker_smoke, and local cert."""

import argparse
import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(mode: str, log_file: str | None):
    lock = threading.Lock()

    class FixtureHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request = json.loads(raw.decode("utf-8"))
            slot_header = self.headers.get("X-Coworld-Player-Slot", "-1")
            user = request["messages"][0]["content"]
            match = re.search(r"SHIFT (\d+)", user)
            shift = int(match.group(1)) if match else 0

            if mode == "infra_error":
                status = 503
                body = json.dumps({"error": {"type": "service_unavailable", "message": "fixture provider outage"}}).encode()
            else:
                status = 200
                if mode == "invalid_first_shift" and shift == 1:
                    text = "not a json order"
                else:
                    # Deterministic valid standing order for any Chemistry variant (amber takes resin+spark).
                    slot = int(slot_header) if slot_header.lstrip("-").isdigit() else 0
                    molecule = "resin" if slot % 2 == 0 else "spark"
                    text = json.dumps({
                        "job": "supply",
                        "molecule": molecule,
                        "reactor": "amber",
                        "say": f"{molecule} to Amber",
                        "notes": f"shift {shift} slot {slot}",
                    })
                body = json.dumps({
                    "id": f"msg_{uuid.uuid4().hex[:12]}",
                    "type": "message",
                    "role": "assistant",
                    "model": request.get("model", "chemistry-fixture"),
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": text}],
                    "usage": {"input_tokens": 120, "output_tokens": 24},
                }).encode()

            if log_file:
                entry = {"path": self.path, "slot": slot_header, "shift": shift, "status": status}
                with lock:
                    with open(log_file, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry) + "\n")

            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Softmax-Llm-Call-Id", str(uuid.uuid5(uuid.NAMESPACE_URL, raw.decode("utf-8"))))
            self.end_headers()
            self.wfile.write(body)

    return FixtureHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--mode", choices=["valid", "infra_error", "invalid_first_shift"], default="valid")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.mode, args.log_file))
    print(server.server_port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
