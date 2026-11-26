# -*- coding: utf-8 -*-
"""Score a chat checkpoint on both corpora, so a chat gain can be checked against
story-continuation damage.

  chat_ce   held-out dialogue, assistant tokens only  (lower = better replies)
  story_ce  held-out TinyStories, all tokens          (lower = better continuations)

Usage: python eval_chat.py <arch> [tag]
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmh_model import StudentLM

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")
ARCH = sys.argv[1] if len(sys.argv) > 1 else "full"
TAG = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""
T, VOCAB, DEV = 512, 50257, "cuda"


def batches(bin_name, mask_name, n_batch=12, bs=8):
    src = np.memmap(os.path.join(DATA, bin_name), dtype=np.uint16, mode="r")
    msk = (np.memmap(os.path.join(DATA, mask_name), dtype=np.uint8, mode="r")
           if mask_name else None)
    rng = np.random.RandomState(0)
    for _ in range(n_batch):
        ix = rng.randint(0, len(src) - T - 1, size=bs)
        x = np.stack([src[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([src[i + 1:i + T + 1] for i in ix]).astype(np.int64)
        m = (np.stack([msk[i + 1:i + T + 1] for i in ix]).astype(np.float32)
             if msk is not None else None)
        yield x, y, m


def score(model, bin_name, mask_name):
    tot, ntok = 0.0, 0.0
    with torch.no_grad():
        for x, y, m in batches(bin_name, mask_name):
            xb, yb = torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(xb).float().view(-1, VOCAB)
            yv = yb.view(-1)
            if m is not None:
                keep = torch.from_numpy(m).to(DEV).view(-1) > 0.5
                lo, yv = lo[keep], yv[keep]
            tot += F.cross_entropy(lo, yv, reduction="sum").item()
            ntok += yv.numel()
    return tot / max(ntok, 1)


def build(ckpt):
    m = StudentLM(VOCAB, d=384, layers=8, heads=4, window=256, kind=ARCH).to(DEV)
    m.load_state_dict(torch.load(os.path.join(OUT, ckpt), map_location=DEV, weights_only=False))
    m.eval()
    return m


rows = []
rows.append(("pretrained", build(f"student_scratch_{ARCH}.pt")))
rows.append((f"sft{TAG or '_base'}", build(f"student_sft_{ARCH}{TAG}.pt")))
print(f"{'model':22s} {'chat_ce':>8s} {'story_ce':>9s} {'chat_ppl':>9s}")
for name, m in rows:
    c = score(m, "val_chat.bin", "val_chat.mask")
    s = score(m, "val.bin", None)
    print(f"{name:22s} {c:8.4f} {s:9.4f} {math.exp(min(c, 20)):9.2f}")
