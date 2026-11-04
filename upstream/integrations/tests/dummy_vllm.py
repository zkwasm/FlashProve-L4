#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for "vllm serve MODEL_DIR --middleware MODULE.CLASS", for tests. It runs no model.

It serves /health and /v1/completions on 127.0.0.1 through the middleware and, except on /health,
requires the API key in VLLM_API_KEY. It ignores vllm serve's other options. A reply counts up from
the prompt's last token ID by "step", to max_tokens tokens or to "eos_token_id";
MODEL_DIR/config.json sets "step" and "eos_token_id". dummy_service.py's stand-in for the committed
model counts up by 1. Each call runs in a new event loop, whereas vLLM has one for all calls. The
script prints its port, and exits when its parent process exits.
"""

import argparse
import asyncio
import http.server
import importlib
import json
import os
import sys

if __name__ == "__main__":  # run by path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0], allow_abbrev=False)
    p.add_argument("command", choices=("serve",))
    p.add_argument("model_dir", metavar="MODEL_DIR")
    p.add_argument(
        "--port", type=int, default=0, help="the port to listen on (default: 0, any free port)"
    )
    p.add_argument("--middleware", required=True, metavar="MODULE.CLASS")
    return p.parse_known_args(argv)[0]


def completion(model, body):
    """Return part of vLLM's response body for a completion."""
    prompt, reply = body["prompt"], []
    while len(reply) < body["max_tokens"] and reply[-1:] != [model["eos_token_id"]]:
        reply.append((reply or prompt)[-1] + model["step"])
    choice = {
        "index": 0,
        "text": "",
        "finish_reason": "stop" if reply[-1] == model["eos_token_id"] else "length",
        "token_ids": reply,
        "prompt_token_ids": prompt,
    }
    return {"object": "text_completion", "model": body["model"], "choices": [choice]}


def inner_application(model, api_key):
    async def application(scope, receive, send):
        status, body = 404, {"error": {"message": "not found"}}
        route = (scope["method"], scope["path"])
        if route == ("GET", "/health"):
            status, body = 200, {}
        elif (b"authorization", b"Bearer " + api_key.encode()) not in scope["headers"]:
            status, body = 401, {"error": {"message": "Unauthorized"}}
        elif route == ("POST", "/v1/completions"):
            status, body = 200, completion(model, json.loads((await receive())["body"]))
        data = json.dumps(body).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(data))]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": data})

    return application


def handler(application):
    """Return a handler that passes HTTP calls to an ASGI application."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def call(self):
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            waiting = [{"type": "http.request", "body": body, "more_body": False}]
            headers = [(k.lower().encode(), v.encode()) for k, v in self.headers.items()]
            scope = {"type": "http", "method": self.command, "path": self.path, "headers": headers}

            async def receive():
                if not waiting:
                    await asyncio.Event().wait()  # this server never reports a disconnect
                return waiting.pop()

            async def send(message):
                if message["type"] == "http.response.start":
                    self.send_response(message["status"])
                    for name, value in message["headers"]:
                        self.send_header(name.decode(), value.decode())
                    self.end_headers()
                else:
                    self.wfile.write(message.get("body", b""))

            asyncio.run(application(scope, receive, send))

        do_GET = do_POST = do_DELETE = call

    return Handler


def main(argv=None):
    args = parse_args(argv)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        model = json.load(f)
    module, _, name = args.middleware.rpartition(".")
    middleware = getattr(importlib.import_module(module), name)
    application = middleware(inner_application(model, os.environ["VLLM_API_KEY"]))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler(application)) as server:
        print("dummy_vllm.py: a stand-in for tests, on port %d" % server.server_port, flush=True)
        server.timeout, parent = 0.05, os.getppid()
        while os.getppid() == parent:
            server.handle_request()
    return 1


if __name__ == "__main__":
    sys.exit(main())
