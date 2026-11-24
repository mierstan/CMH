# -*- coding: utf-8 -*-
"""CMH (Compressed-Memory Hybrid) prototype experiments.
Writes all results to result.json / log.txt (stdout may not echo on this box).
"""
import math, time, json, random, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

torch.manual_seed(0); random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

D, HEADS, LAYERS, WIN, NSLOT = 128, 4, 4, 128, 128
RES = {}

# ---------------- attention kernels ----------------
def attn(q, k, v, mask=None):
    h = q.shape[2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None: scores = scores.masked_fill(mask, float("-inf"))
    return (F.softmax(scores, dim=-1) @ v)

def split(x, h):
    B, T, D = x.shape
    return x.view(B, T, h, D // h).transpose(1, 2)  # [B,h,T,dh]

class WindowSelfAttn(nn.Module):
    """True sliding-window causal attention: each chunk of w queries attends to
    itself plus the previous ctx tokens (stride w, effective window w+ctx)."""
    def __init__(self, d, h, w, ctx=None):
        super().__init__()
        self.h, self.w = h, w
        self.ctx = ctx or w
        self.qk = nn.Linear(d, 2 * d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
    def attend(self, cur, prev):
        # cur: [B,t,D] queries; prev: [B,p,D] attend-only context (all before cur)
        B, t, D = cur.shape
        seg = cur if prev is None else torch.cat([prev, cur], dim=1)
        p = seg.shape[1] - t
        q, _ = self.qk(cur).chunk(2, dim=-1)
        _, k = self.qk(seg).chunk(2, dim=-1)
        v = self.v(seg)
        dev = cur.device
        u = torch.arange(p + t, device=dev).unsqueeze(0)          # key global idx
        qi = p + torch.arange(t, device=dev).unsqueeze(1)         # query global idx
        mask = u > qi                                              # blocked if key after query
        dh = D // self.h
        def sp(x): return x.view(B, -1, self.h, dh).transpose(1, 2)
        out = attn(sp(q), sp(k), sp(v), mask[None, None])            # [B,h,t,dh]
        out = out.transpose(1, 2).reshape(B, t, D)
        return self.o(out)
    def forward(self, x):
        B, T, D = x.shape
        outs, s = [], 0
        prev = None
        while s < T:
            cur = x[:, s:s + self.w]
            outs.append(self.attend(cur, prev))
            e = min(s + self.w, T)
            prev = x[:, max(0, e - self.ctx):e]
            s += self.w
        return torch.cat(outs, dim=1)

# ---------------- blocks ----------------
class FullBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.qk = nn.Linear(d, 2 * d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h_ = h
    def forward(self, x, mem=None):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        q, k = self.qk(self.ln(x)).chunk(2, dim=-1)
        a = attn(split(q, self.h_), split(k, self.h_), split(self.v(self.ln(x)), self.h_), mask)
        x = x + self.o(a.reshape(x.shape[0], T, -1))
        return x + self.f(x), mem

class WindowBlock(nn.Module):
    def __init__(self, d, h, w):
        super().__init__()
        self.attn = WindowSelfAttn(d, h, w)
        self.ln = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, mem=None):
        x = x + self.attn(self.ln(x))
        return x + self.f(x), mem

class CMHBlock(nn.Module):
    """Chunked-recurrent hybrid: per chunk -> associative-memory read (o=qS/qn),
    window self-attention, FFN, gated associative write (S <- gS + k^T v).
    Fully causal; memory is an outer-product key-value state, so written tokens
    keep their own keys and are addressable by content/position."""
    def __init__(self, d, h, w):
        super().__init__()
        self.attn = WindowSelfAttn(d, h, w)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)
        self.rq = nn.Linear(d, d); self.ro = nn.Linear(d, d)          # read
        self.wk = nn.Linear(d, d); self.wv = nn.Linear(d, d)          # write
        self.wg = nn.Linear(d, 1)                                     # per-token salience gate
        self.decay = nn.Parameter(torch.full((h,), 3.0))              # per-head retention ~0.95/chunk
        self.h_ = h; self.w = w
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, mem):
        B, T, D = x.shape; dh = D // self.h_
        S = x.new_zeros(B, self.h_, dh, dh); n = x.new_zeros(B, self.h_, 1, dh)
        g = torch.sigmoid(self.decay).view(1, self.h_, 1, 1)
        out = []
        prev_h = None
        for s in range(0, T, self.w):
            xc = x[:, s:s + self.w]
            # 1) read from associative memory (state holds only prior chunks)
            q = split(self.rq(self.ln2(xc)), self.h_)                 # [B,h,t,dh]
            o = (q @ S) / ((q @ n.transpose(-2, -1)) + 1e-6)          # [B,h,t,dh]
            o = o.transpose(1, 2).reshape(B, -1, D)
            h_ = xc + self.ro(o)
            # 2) sliding-window local attention (prev chunk = context) + 3) FFN
            h_ = h_ + self.attn.attend(self.ln1(h_), None if prev_h is None else self.ln1(prev_h))
            h_ = h_ + self.f(h_)
            prev_h = h_
            # 4) gated associative write
            k = split(self.wk(self.ln3(h_)), self.h_)                 # [B,h,t,dh]
            v = split(self.wv(self.ln3(h_)), self.h_)
            gt = torch.sigmoid(self.wg(self.ln3(h_))).unsqueeze(1)    # [B,1,t,1]
            k = k * gt
            S = g * S + k.transpose(-2, -1) @ v                       # [B,h,dh,dh]
            n = g * n + k.sum(dim=2, keepdim=True)                    # [B,h,1,dh]
            out.append(h_)
        return torch.cat(out, dim=1), mem

class Model(nn.Module):
    def __init__(self, vocab, kind):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.pos = nn.Parameter(torch.randn(1, 4096, D) * 0.02)
        self.kind = kind
        if kind == "full":   blocks = [FullBlock(D, HEADS) for _ in range(LAYERS)]
        elif kind == "win":  blocks = [WindowBlock(D, HEADS, WIN) for _ in range(LAYERS)]
        else:                blocks = [CMHBlock(D, HEADS, WIN) for _ in range(LAYERS)]
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(D); self.head = nn.Linear(D, vocab)
        self.reset_mem()
    def reset_mem(self):
        self.mem = [None] * LAYERS
    def forward(self, idx):
        x = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for i, b in enumerate(self.blocks):
            x, self.mem[i] = b(x, self.mem[i])
        return self.head(self.ln(x))
    def fresh(self, B):
        self.reset_mem()
        return self

# ---------------- LM task: synthetic structured text ----------------
def make_lm_corpus(n_stmt=60000):
    vars_ = []
    lines = []
    for _ in range(n_stmt):
        if vars_ and random.random() < 0.5:
            a, b = random.choice(vars_), random.choice(vars_)
        else:
            a = "v%d" % random.randint(0, 99); vars_.append(a)
            b = "v%d" % random.randint(0, 99); vars_.append(b)
        tgt = "v%d" % random.randint(0, 99)
        lines.append("%s = %s + %s ;" % (tgt, a, b))
        if vars_ and random.random() < 0.4:
            lines.append("use %s ;" % random.choice(vars_))
    toks = []
    for l in lines: toks += l.split() + ["\n"]
    vocab = {t: i + 1 for i, t in enumerate(sorted(set(toks)))}
    ids = [vocab[t] for t in toks]
    return ids, vocab

def train_lm(kind, data, vocab_n, steps=2000, B=32, T=256):
    model = Model(vocab_n, kind).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tr = data[: int(len(data) * 0.9)]; va = data[int(len(data) * 0.9):]
    def batch(src):
        i = random.randint(0, len(src) - T - 1)
        x = torch.tensor(src[i:i + T], device=DEV).unsqueeze(0).expand(B, T).contiguous()
        return x
    t0 = time.time()
    for s in range(steps):
        x = batch(tr)
        model.fresh(B)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_n), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 300 == 0 or s == steps - 1:
            model.fresh(B)
            with torch.no_grad():
                xv = batch(va)
                lv = F.cross_entropy(model(xv)[:, :-1].reshape(-1, vocab_n), xv[:, 1:].reshape(-1))
            log(f"  LM[{kind}] step {s}: train {loss.item():.4f} val {lv.item():.4f}")
    dt = time.time() - t0
    model.fresh(B)
    with torch.no_grad():
        xv = batch(va); lv = F.cross_entropy(model(xv)[:, :-1].reshape(-1, vocab_n), xv[:, 1:].reshape(-1)).item()
    p = sum(p.numel() for p in model.parameters())
    log(f"LM[{kind}] final val {lv:.4f}  params {p}  train_time {dt:.1f}s")
    del model; torch.cuda.empty_cache()
    return {"val_loss": lv, "params": p, "train_sec": dt}

# ---------------- Copy task ----------------
def make_copy_batch(B=16, V=32, pre=64, T=512):
    x = torch.zeros(B, T, dtype=torch.long)
    x[:, :pre] = torch.randint(1, V, (B, pre))
    x[:, T - pre:] = x[:, :pre]          # echo: must re-emit prefix at the end
    tgt = torch.cat([x[:, 1:], torch.zeros(B, 1, dtype=torch.long)], dim=1)
    return x.to(DEV), tgt.to(DEV)

def train_copy(kind, steps=8000, B=16):
    model = Model(32, kind).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t0 = time.time()
    for s in range(steps):
        x, tgt = make_copy_batch(B)
        model.fresh(B)
        logits = model(x)
        # upweight echo-recall positions (447..510 predict prefix) 8x
        w = torch.ones(1, 511, device=DEV); w[:, 447:511] = 8.0
        per = F.cross_entropy(logits[:, :-1].reshape(-1, 32), tgt[:, :-1].reshape(-1), reduction="none").view(B, -1)
        loss = (per * w).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 500 == 0: log(f"  COPY[{kind}] step {s}: loss {loss.item():.4f}")
    dt = time.time() - t0
    # eval: accuracy on final 64 predicted positions
    model.fresh(B); accs = []
    with torch.no_grad():
        for _ in range(20):
            x, _ = make_copy_batch(16)
            pred = model(x).argmax(-1)
            accs.append((pred[:, 447:511] == x[:, 448:512]).float().mean().item())
    acc = sum(accs) / len(accs)
    log(f"COPY[{kind}] recall_acc {acc:.4f}  train_time {dt:.1f}s")
    del model; torch.cuda.empty_cache()
    return {"recall_acc": acc, "train_sec": dt}

# ---------------- Efficiency benchmark ----------------
def bench(kind, T, B=8, iters=10):
    model = Model(1000, kind).to(DEV)
    x = torch.randint(1, 1000, (B, T), device=DEV)
    model.fresh(B)
    for _ in range(3):
        model.fresh(B)
        loss = model(x).float().pow(2).mean(); loss.backward()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        model.fresh(B)
        loss = model(x).float().pow(2).mean()
        opt0 = [p.grad.zero_() for p in model.parameters() if p.grad is not None]
        loss.backward()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 1e9
    tps = B * T / dt
    log(f"BENCH[{kind}] T={T}: {tps:.0f} tok/s  peak {peak:.2f} GB")
    del model, x; torch.cuda.empty_cache()
    return {"tokens_per_sec": tps, "peak_gb": peak}

# ---------------- run all ----------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"

if MODE in ("all", "lm"):
    ids, vocab = make_lm_corpus()
    log("corpus tokens:", len(ids), "vocab:", len(vocab) + 1)
    RES["lm"] = {}
    for kind in ["full", "win", "cmh"]:
        RES["lm"][kind] = train_lm(kind, ids, len(vocab) + 1)

if MODE in ("all", "copy"):
    RES["copy"] = {}
    for kind in ["full", "win", "cmh"]:
        RES["copy"][kind] = train_copy(kind)

if MODE in ("all", "bench"):
    RES["bench"] = {}
    for kind in ["full", "cmh"]:
        RES["bench"][kind] = {T: bench(kind, T) for T in [256, 512, 1024, 2048, 4096]}

with open(os.path.join(OUT, f"result_{MODE}.json"), "w", encoding="utf-8") as f:
    json.dump(RES, f, indent=2)
log("ALL DONE")
