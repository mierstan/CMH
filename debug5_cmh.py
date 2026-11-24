# -*- coding: utf-8 -*-
"""debug5: capability curve - cmh vs win across gaps (fresh batches)."""
import sys, os, math, random
import torch
import torch.nn.functional as F

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmh_experiments.py")
src = open(SRC, encoding="utf-8").read().split("# ---------------- run all ----------------")[0]
exec(src)

def fresh_batch(B, pre, gap):
    T = 2 * pre + gap
    x = torch.zeros(B, T, dtype=torch.long)
    x[:, :pre] = torch.randint(1, 32, (B, pre))
    x[:, pre + gap:] = x[:, :pre]
    tgt = torch.cat([x[:, 1:], torch.zeros(B, 1, dtype=torch.long)], dim=1)
    return x.to(DEV), tgt.to(DEV), T

def run(kind, gap, steps=3000, B=16):
    model = Model(32, kind).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for s in range(steps):
        x, tgt, T = fresh_batch(B, 64, gap)
        model.fresh(B)
        logits = model(x)
        w = torch.ones(1, T - 1, device=DEV); w[:, -64:] = 8.0
        per = F.cross_entropy(logits[:, :-1].reshape(-1, 32), tgt[:, :-1].reshape(-1), reduction="none").view(B, -1)
        loss = (per * w).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    accs = []
    with torch.no_grad():
        for _ in range(20):
            x, tgt, T = fresh_batch(B, 64, gap)
            model.fresh(B)
            pred = model(x).argmax(-1)
            accs.append((pred[:, T - 65:T - 1] == x[:, T - 64:]).float().mean().item())
    log(f"D5 {kind} gap={gap} steps={steps}: acc={sum(accs)/len(accs):.4f}")
    del model; torch.cuda.empty_cache()

for gap in [96, 192, 256, 320]:
    run("cmh", gap)
for gap in [192, 256]:
    run("win", gap)
log("DEBUG5 DONE")
