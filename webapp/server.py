# -*- coding: utf-8 -*-
"""Demo server for the trained students: chat mode + story-continuation mode.
Serves webapp/index.html, /api/chat and /api/generate.
Run: python server.py  ->  http://127.0.0.1:8765
"""
import json, os, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import GPT2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmh_model import StudentLM
from chat_sample import sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # distill/
WEBAPP = os.path.join(ROOT, "webapp")
PORT = 8765
DEV = "cuda"
VOCAB = 50257
CTX = 512

tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "models", "gpt2"))
print("loading models ...", flush=True)
MODELS = {}
SPECS = [
    ("chat-full", "student_sft_full_v5.pt"),   # replay + persona + generic follow-ups
    ("chat-cmh", "student_sft_cmh.pt"),
    ("full", "student_scratch_full.pt"),
    ("cmh", "student_scratch_cmh.pt"),
]
for name, ckpt in SPECS:
    path = os.path.join(ROOT, "out", ckpt)
    if not os.path.exists(path):
        print(f"  {name}: {ckpt} not found, skipped", flush=True)
        continue
    kind = "full" if name.endswith("full") else "cmh"
    m = StudentLM(VOCAB, d=384, layers=8, heads=4, window=256, kind=kind).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV, weights_only=False))
    m.eval()
    MODELS[name] = m
    print(f"  {name} ready", flush=True)

INDEX = open(os.path.join(WEBAPP, "index.html"), encoding="utf-8").read()


def gen_from(prompt_ids, model, max_new, temperature, top_k, chat=False):
    """chat=True uses the tuned sampler: repetition controls plus a stop at the
    training-time turn separator, which is where the model means to end its reply."""
    if chat:
        ids = sample(model, prompt_ids, max_new, temperature=temperature, top_k=top_k,
                     top_p=0.9, rep_penalty=1.2, no_repeat_ngram=3,
                     stop_sequences=[[628], [198, 198]])
        return tok.decode(ids).strip().replace("<|endoftext|>", " ")
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new, temperature=temperature, top_k=top_k)
    text = tok.decode(out)
    for stop_at in ("\nUser:", "\n\n"):
        if stop_at in text:
            text = text[:text.index(stop_at)]
    return text.replace("<|endoftext|>", " ")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")     # never serve a stale UI
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
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            model = req.get("model", "full")
            max_new = min(int(req.get("max_new", 120)), 300)
            temperature = float(req.get("temperature", 0.85))
            top_k = int(req.get("top_k", 40))
            if model not in MODELS:
                return self._send(400, json.dumps({"error": "unknown model"}))
            if self.path == "/api/chat":
                history = (req.get("history") or req.get("transcript") or "")
                if not isinstance(history, str):
                    history = ""
                history = history.strip()
                if not history:
                    print(f"[chat] empty history, keys={list(req)}", flush=True)
                    return self._send(400, json.dumps({"error": "empty history"}))
                # Model was trained on "Assistant:" with no trailing space: GPT-2 BPE
                # would otherwise emit a lone ' ' token (id 220) and generation derails.
                history = re.sub(r":\s+$", ":", history)
                if not history.endswith("Assistant:"):
                    history = history.rstrip() + "\n\nAssistant:"
                ids = tok(history, add_special_tokens=False)["input_ids"]
                ids = ids[-(CTX - 140):]                     # leave room for generation
                ids = torch.tensor([ids], dtype=torch.long).cuda()
                text = gen_from(ids, MODELS[model], max_new, temperature, top_k, chat=True)
                return self._send(200, json.dumps({"text": text.strip()}))
            if self.path == "/api/generate":
                prompt = (req.get("prompt") or "").strip() or "Once upon a time,"
                ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
                text = gen_from(ids, MODELS[model], max_new, temperature, top_k)
                return self._send(200, json.dumps({"text": text}))
            return self._send(404, json.dumps({"error": "not found"}))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"serving at http://127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()
