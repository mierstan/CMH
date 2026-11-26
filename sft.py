# -*- coding: utf-8 -*-
"""Chat SFT: fine-tune a trained student into a conversational model.
Masked CE only on assistant spans. Warm start from an existing checkpoint.
Usage: python sft.py <arch> [steps]     e.g. python sft.py full 4000
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmh_model import StudentLM, param_count

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
ARCH = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

B, T = 24, 512
LR = 1e-4                               # low LR: avoid catastrophic forgetting
WARMUP = 150
EVAL_EVERY = 500
DEV = "cuda"
VOCAB = 50257
WARM = os.path.join(OUT, f"student_scratch_{ARCH}.pt")

log_f = open(os.path.join(OUT, f"sft_{ARCH}.log"), "a", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    log_f.write(s + "\n"); log_f.flush()

torch.manual_seed(0); np.random.seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

train = np.memmap(os.path.join(ROOT, "data", "train_chat.bin"), dtype=np.uint16, mode="r")
tmask = np.memmap(os.path.join(ROOT, "data", "train_chat.mask"), dtype=np.uint8, mode="r")
val = np.memmap(os.path.join(ROOT, "data", "val_chat.bin"), dtype=np.uint16, mode="r")
vmask = np.memmap(os.path.join(ROOT, "data", "val_chat.mask"), dtype=np.uint8, mode="r")

def get_batch(src, msk, B, T):
    ix = np.random.randint(0, len(src) - T - 1, size=B)
    x = np.stack([src[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([src[i + 1:i + T + 1] for i in ix]).astype(np.int64)
    m = np.stack([msk[i + 1:i + T + 1] for i in ix]).astype(np.float32)
    return (torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV),
            torch.from_numpy(m).to(DEV))

model = StudentLM(VOCAB, d=384, layers=8, heads=4, window=256, kind=ARCH).to(DEV)
model.load_state_dict(torch.load(WARM, map_location=DEV, weights_only=False))
log(f"SFT {ARCH} from {os.path.basename(WARM)} steps={STEPS} params={param_count(model)/1e6:.1f}M")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS,
                                            pct_start=min(0.3, max(0.02, WARMUP / STEPS)),
                                            anneal_strategy="cos")

CKPT = os.path.join(OUT, f"ckpt_sft_{ARCH}.pt")
start_step = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    start_step = ck["step"]
    log(f"RESUMED from step {start_step}")

def save_ckpt(step):
    tmp = CKPT + ".tmp"
    torch.save({"step": step, "model": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict()}, tmp)
    os.replace(tmp, CKPT)

def eval_val():
    model.eval()
    losses, ntok = 0.0, 0.0
    with torch.no_grad():
        for _ in range(12):
            x, y, m = get_batch(val, vmask, 8, T)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(x)
            lo = lo.float().view(-1, VOCAB)
            yv, mv = y.view(-1), m.view(-1)
            sel = mv > 0.5
            if sel.sum() == 0:
                continue
            losses += F.cross_entropy(lo[sel], yv[sel], reduction="sum").item()
            ntok += sel.sum().item()
    model.train()
    return losses / max(ntok, 1)

t0 = time.time()
for step in range(start_step, STEPS):
    x, y, m = get_batch(train, tmask, B, T)
    opt.zero_grad()
    loss_sum = 0.0
    for mi in range(4):                              # micro-batch 6
        xm, ym, mm = x[mi * 6:(mi + 1) * 6], y[mi * 6:(mi + 1) * 6], m[mi * 6:(mi + 1) * 6]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(xm)
        lo = lo.float().view(-1, VOCAB)
        yv, mv = ym.reshape(-1), mm.reshape(-1)
        sel = mv > 0.5
        loss = F.cross_entropy(lo[sel], yv[sel]) if sel.sum() > 0 else lo.mean() * 0
        (loss / 4).backward()
        loss_sum += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 200 == 0 or step == STEPS - 1:
        log(f"step {step}: loss {loss_sum/4:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")
    if (step + 1) % EVAL_EVERY == 0 or step == STEPS - 1:
        vl = eval_val()
        log(f"== step {step+1}: val_asst_ce {vl:.4f} (ppl {math.exp(min(vl,20)):.2f})")
        save_ckpt(step + 1)

torch.save(model.state_dict(), os.path.join(OUT, f"student_sft_{ARCH}.pt"))
log(f"DONE sft {ARCH} in {time.time()-t0:.0f}s")
