# -*- coding: utf-8 -*-
"""Story-continuation demo server for the trained students.
Serves webapp/index.html and a /api/generate endpoint.
Run: python server.py  ->  http://127.0.0.1:8765
"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import GPT2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmh_model import StudentLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # distill/
WEBAPP = os.path.join(ROOT, "webapp")
PORT = 8765
DEV = "cuda"
VOCAB = 50257

tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "models", "gpt2"))
print("loading models ...", flush=True)
MODELS = {}
for kind, ckpt in [("full", "student_scratch_full.pt"), ("cmh", "student_scratch_cmh.pt")]:
    m = StudentLM(VOCAB, d=384, layers=8, heads=4, window=256, kind=kind).to(DEV)
    m.load_state_dict(torch.load(os.path.join(ROOT, "out", ckpt), map_location=DEV, weights_only=False))
    m.eval()
    MODELS[kind] = m
    print(f"  {kind} ready", flush=True)

INDEX = open(os.path.join(WEBAPP, "index.html"), encoding="utf-8").read()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, INDEX, "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True, "models": list(MODELS)}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/generate":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            prompt = (req.get("prompt") or "").strip() or "Once upon a time,"
            model = req.get("model", "full")
            max_new = min(int(req.get("max_new", 120)), 300)
            temperature = float(req.get("temperature", 0.85))
            top_k = int(req.get("top_k", 40))
            if model not in MODELS:
                return self._send(400, json.dumps({"error": "unknown model"}))
            ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
            with torch.no_grad():
                out = MODELS[model].generate(ids, max_new, temperature=temperature, top_k=top_k)
            text = tok.decode(out)
            self._send(200, json.dumps({"text": text}))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"serving at http://127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()
