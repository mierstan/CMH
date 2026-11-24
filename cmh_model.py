# -*- coding: utf-8 -*-
"""Student architectures for the distillation study: CMH (sliding-window attention
+ chunked-recurrent associative memory), plus full-attention and window-only baselines.
All share the same tokenizer, depth budget, and training loop (see distill.py)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _attn(q, k, v, mask=None):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def _split(x, h):
    B, T, D = x.shape
    return x.view(B, T, h, D // h).transpose(1, 2)


class SlidingWindowAttn(nn.Module):
    """True sliding-window causal attention (chunk queries + attend-only ctx)."""
    def __init__(self, d, h, w, ctx=None):
        super().__init__()
        self.h, self.w = h, w
        self.ctx = ctx if ctx is not None else w
        self.qk = nn.Linear(d, 2 * d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def attend(self, cur, prev):
        B, t, D = cur.shape
        seg = cur if prev is None else torch.cat([prev, cur], dim=1)
        p = seg.shape[1] - t
        q, _ = self.qk(cur).chunk(2, dim=-1)
        _, k = self.qk(seg).chunk(2, dim=-1)
        v = self.v(seg)
        u = torch.arange(p + t, device=cur.device).unsqueeze(0)
        qi = p + torch.arange(t, device=cur.device).unsqueeze(1)
        mask = (u > qi)[None, None]
        dh = D // self.h
        out = _attn(_split(q, self.h), _split(k, self.h), _split(v, self.h), mask)
        return self.o(out.transpose(1, 2).reshape(B, t, D))

    def forward(self, x):
        T = x.shape[1]
        outs, prev, s = [], None, 0
        while s < T:
            cur = x[:, s:s + self.w]
            outs.append(self.attend(cur, prev))
            e = min(s + self.w, T)
            prev = x[:, max(0, e - self.ctx):e]
            s += self.w
        return torch.cat(outs, dim=1)


class FullBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h_ = h
        self.qk = nn.Linear(d, 2 * d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln3 = nn.LayerNorm(d)

    def forward(self, x):
        B, T, D = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        q, k = self.qk(self.ln1(x)).chunk(2, dim=-1)
        a = _attn(_split(q, self.h_), _split(k, self.h_), _split(self.v(self.ln1(x)), self.h_), mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.f(self.ln2(x))


class WindowBlock(nn.Module):
    def __init__(self, d, h, w):
        super().__init__()
        self.attn = SlidingWindowAttn(d, h, w)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.f(self.ln2(x))


class CMHBlock(nn.Module):
    """Chunked-recurrent hybrid: per chunk -> associative-memory read,
    sliding-window attention (prev chunk as ctx), FFN, gated associative write."""
    def __init__(self, d, h, w):
        super().__init__()
        self.attn = SlidingWindowAttn(d, h, w, ctx=w)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.rq = nn.Linear(d, d)
        self.ro = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wg = nn.Linear(d, 1)
        self.decay = nn.Parameter(torch.full((h,), 3.0))
        self.h_ = h
        self.w = w
        self.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        dh = D // self.h_
        S = x.new_zeros(B, self.h_, dh, dh)
        n = x.new_zeros(B, self.h_, 1, dh)
        g = torch.sigmoid(self.decay).view(1, self.h_, 1, 1)
        out, prev_h = [], None
        for s in range(0, T, self.w):
            xc = x[:, s:s + self.w]
            q = _split(self.rq(self.ln2(xc)), self.h_)
            o = (q @ S) / ((q @ n.transpose(-2, -1)) + 1e-6)
            o = o.transpose(1, 2).reshape(B, -1, D)
            h_ = xc + self.ro(o)
            h_ = h_ + self.attn.attend(self.ln1(h_),
                                       None if prev_h is None else self.ln1(prev_h))
            h_ = h_ + self.f(self.ln3(h_))
            prev_h = h_
            k = _split(self.wk(self.ln3(h_)), self.h_)
            v = _split(self.wv(self.ln3(h_)), self.h_)
            gt = torch.sigmoid(self.wg(self.ln3(h_))).unsqueeze(1)
            k = k * gt
            S = g * S + k.transpose(-2, -1) @ v
            n = g * n + k.sum(dim=2, keepdim=True)
            out.append(h_)
        return torch.cat(out, dim=1)


class StudentLM(nn.Module):
    def __init__(self, vocab, d=384, layers=8, heads=4, window=256, kind="cmh",
                 max_len=512, tie=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        if kind == "full":
            blocks = [FullBlock(d, heads) for _ in range(layers)]
        elif kind == "win":
            blocks = [WindowBlock(d, heads, window) for _ in range(layers)]
        else:
            blocks = [CMHBlock(d, heads, window) for _ in range(layers)]
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        if tie:
            self.head.weight = self.emb.weight
        self.apply(self._init)
        with torch.no_grad():
            self.emb.weight.normal_(0, 0.02)
            if tie:
                self.head.weight = self.emb.weight  # re-tie after init
                self.head.weight.normal_(0, 0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, prompt_ids, n_new, temperature=0.9, top_k=50):
        self.eval()
        x = prompt_ids.clone()
        for _ in range(n_new):
            logits = self(x[:, -self.pos.shape[1]:])[0, -1]
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[-1]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            x = torch.cat([x, nxt.view(1, 1)], dim=1)
        self.train()
        return x[0, prompt_ids.shape[1]:].tolist()


def param_count(m):
    return sum(p.numel() for p in m.parameters())
