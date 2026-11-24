# -*- coding: utf-8 -*-
"""Three cheap follow-up tests on the trained students:
1. CMH with memory disabled at inference (does the state contribute anything?)
2. Learned per-head retention gamma
3. Zero-shot copy-recall of TinyStories-trained students
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as F
from cmh_model import StudentLM, CMHBlock, SlidingWindowAttn, _split, _attn
import cmh_model

DEV = "cuda"
VOCAB = 50257
OUT = os.path.dirname(os.path.abspath(__file__))
rep = open(os.path.join(OUT, "extra_tests.txt"), "w", encoding="utf-8")
def log(*a):
    rep.write(" ".join(str(x) for x in a) + "\n"); rep.flush()

val = np.memmap(os.path.join(OUT, "data", "val.bin"), dtype=np.uint16, mode="r")
def batch(B=8, T=512):
    ix = np.random.randint(0, len(val) - T - 1, size=B)
    x = np.stack([val[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([val[i + 1:i + T + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)

def val_ce(m, disable_memory=False):
    losses = []
    with torch.no_grad():
        for _ in range(16):
            x, y = batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(x)
            losses.append(F.cross_entropy(lo.view(-1, VOCAB).float(), y.view(-1)).item())
    return sum(losses) / len(losses)

# ---- 1. memory ablation at inference ----
m = StudentLM(VOCAB, kind="cmh").to(DEV)
m.load_state_dict(torch.load(os.path.join(OUT, "out", "student_scratch_cmh.pt"),
                             map_location=DEV, weights_only=False))
m.eval()
ce_normal = val_ce(m)
# monkeypatch: skip the associative read and write, keep window+ffn
orig_forward = CMHBlock.forward
def blind_forward(self, x):
    B, T, D = x.shape
    out, prev_h = [], None
    for s in range(0, T, self.w):
        h_ = x[:, s:s + self.w]
        h_ = h_ + self.attn.attend(self.ln1(h_),
                                   None if prev_h is None else self.ln1(prev_h))
        h_ = h_ + self.f(self.ln3(h_))
        prev_h = h_
        out.append(h_)
    return torch.cat(out, dim=1)
CMHBlock.forward = blind_forward
ce_blind = val_ce(m)
CMHBlock.forward = orig_forward
log(f"[ablation] CMH val_ce normal {ce_normal:.4f} | memory-disabled {ce_blind:.4f} | delta {ce_blind - ce_normal:+.4f}")

# ---- 2. learned retention ----
for i, b in enumerate(m.blocks):
    g = torch.sigmoid(b.decay).detach().cpu().tolist()
    log(f"[gamma] layer {i}: " + ", ".join(f"{v:.3f}" for v in g))
del m; torch.cuda.empty_cache()

# ---- 3. zero-shot copy recall of TinyStories-trained students ----
def fresh_copy(B, pre, gap, V=50257):
    T = 2 * pre + gap
    x = torch.zeros(B, T, dtype=torch.long)
    x[:, :pre] = torch.randint(1000, V, (B, pre))   # rare tokens, like real words
    x[:, pre + gap:] = x[:, :pre]
    return x.to(DEV), T

for kind, ck in [("full", "student_scratch_full.pt"), ("cmh", "student_scratch_cmh.pt")]:
    m = StudentLM(VOCAB, kind=kind).to(DEV)
    m.load_state_dict(torch.load(os.path.join(OUT, "out", ck), map_location=DEV, weights_only=False))
    m.eval()
    for gap in [64, 256, 384]:
        accs = []
        with torch.no_grad():
            for _ in range(10):
                x, T = fresh_copy(8, 64, gap)
                pred = m(x).argmax(-1)
                accs.append((pred[:, T - 65:T - 1] == x[:, T - 64:]).float().mean().item())
        log(f"[copy-zero-shot] {kind} gap={gap}: acc={sum(accs)/len(accs):.4f}")
    del m; torch.cuda.empty_cache()
log("EXTRA TESTS DONE")
