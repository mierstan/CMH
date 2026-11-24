# -*- coding: utf-8 -*-
"""Tokenize TinyStories text with GPT-2 tokenizer into a uint16 bin + val bin."""
import os, sys
import numpy as np
from transformers import GPT2TokenizerFast

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(DATA, "TinyStories-train.txt")
MAX_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 400   # text MB to use

tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "models", "gpt2"))

ids = []
total_bytes = 0
CHUNK = 4 * 1024 * 1024
with open(SRC, "r", encoding="utf-8", errors="ignore") as f:
    while total_bytes < MAX_MB * 1024 * 1024:
        block = f.read(CHUNK)
        if not block:
            break
        total_bytes += len(block.encode("utf-8", errors="ignore"))
        # split on story separator to align with story boundaries
        parts = block.split("<|endoftext|>")
        texts = [p.strip() for p in parts if p.strip()]
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for e in enc:
            e.append(50256)  # <|endoftext|> between stories
            ids.extend(e)
        print(f"{total_bytes/1048576:.0f} MB -> {len(ids)/1e6:.1f}M tokens", flush=True)

arr = np.array(ids, dtype=np.uint16)
n_val = int(len(arr) * 0.02)
arr[:-n_val].tofile(os.path.join(DATA, "train.bin"))
arr[-n_val:].tofile(os.path.join(DATA, "val.bin"))
print(f"DONE total {len(arr)/1e6:.1f}M tokens, val {n_val/1e6:.1f}M")
