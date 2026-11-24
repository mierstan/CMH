# -*- coding: utf-8 -*-
"""Curriculum diagnostic: grow the echo gap gradually, fresh batches."""
import sys, os, math, random, time
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

def evaluate(model, gap, B=16):
    accs = []
    with torch.no_grad():
        for _ in range(20):
            x, tgt, T = fresh_batch(B, 64, gap)
            model.fresh(B)
            pred = model(x).argmax(-1)
            accs.append((pred[:, T - 65:T - 1] == x[:, T - 64:]).float().mean().item())
    return sum(accs) / len(accs)

model = Model(32, "cmh").to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
GAPS = [32, 96, 160, 224, 288, 352, 384]
PHASE = 700
for gap in GAPS:
    for s in range(PHASE):
        x, tgt, T = fresh_batch(16, 64, gap)
        model.fresh(16)
        logits = model(x)
        w = torch.ones(1, T - 1, device=DEV); w[:, -64:] = 8.0
        per = F.cross_entropy(logits[:, :-1].reshape(-1, 32), tgt[:, :-1].reshape(-1), reduction="none").view(16, -1)
        loss = (per * w).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    log(f"CURRICULUM gap={gap}: eval_acc={evaluate(model, gap):.4f}")

log(f"FINAL gap=384 eval_acc={evaluate(model, 384):.4f}")
# also check transfer to unseen longer gap
log(f"TRANSFER gap=512 eval_acc={evaluate(model, 512):.4f}")
log("DEBUG3 DONE")
