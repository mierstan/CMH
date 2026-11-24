# -*- coding: utf-8 -*-
"""Distillation training: student (CMH / window / full) <- GPT-2 teacher.
Modes: distill (KL+CE) | scratch (CE only).
Usage: python distill.py <mode> <arch> [steps]
  e.g.  python distill.py distill cmh 6000
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmh_model import StudentLM, param_count

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
MODE, ARCH = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000

B, T = 24, 512
LR = 6e-4
WARMUP = 300
TEMP, ALPHA = 2.0, 0.5          # KD temperature, CE weight
EVAL_EVERY = 500
DEV = "cuda"
VOCAB = 50257

log_f = open(os.path.join(OUT, f"{MODE}_{ARCH}.log"), "a", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    log_f.write(s + "\n"); log_f.flush()

torch.manual_seed(0); np.random.seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

train = np.memmap(os.path.join(ROOT, "data", "train.bin"), dtype=np.uint16, mode="r")
val = np.memmap(os.path.join(ROOT, "data", "val.bin"), dtype=np.uint16, mode="r")

def get_batch(src, B, T):
    ix = np.random.randint(0, len(src) - T - 1, size=B)
    x = np.stack([src[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([src[i + 1:i + T + 1] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV))

TEACHER_CKPT = os.environ.get("TEACHER_CKPT")
if TEACHER_CKPT:
    # in-domain teacher: a trained full-attention student
    teacher = StudentLM(VOCAB, d=384, layers=8, heads=4, window=256, kind="full").to(DEV).eval()
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEV, weights_only=False))
    log(f"TEACHER: in-domain student ckpt {TEACHER_CKPT}")
else:
    teacher = GPT2LMHeadModel.from_pretrained(os.path.join(ROOT, "models", "gpt2")).to(DEV).eval()
    log("TEACHER: GPT-2 (124M)")
for p in teacher.parameters():
    p.requires_grad_(False)

student = StudentLM(VOCAB, d=384, layers=8, heads=4, window=256, kind=ARCH).to(DEV)

opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS,
                                            pct_start=min(0.3, max(0.02, WARMUP / STEPS)),
                                            anneal_strategy="cos")

CKPT = os.path.join(OUT, f"ckpt_{MODE}_{ARCH}.pt")
start_step = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    student.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    start_step = ck["step"]
    log(f"RESUMED {MODE} {ARCH} from step {start_step}")
else:
    log(f"RUN {MODE} {ARCH} steps={STEPS} params={param_count(student)/1e6:.1f}M B={B} T={T}")

def save_ckpt(step):
    tmp = CKPT + ".tmp"
    torch.save({"step": step, "model": student.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict()}, tmp)
    os.replace(tmp, CKPT)

def teacher_forward(xb):
    lo = teacher(xb)
    return lo if TEACHER_CKPT else lo.logits

def eval_val():
    student.eval()
    losses = []
    with torch.no_grad():
        for _ in range(12):
            x, y = get_batch(val, 8, T)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = student(x)
            losses.append(F.cross_entropy(lo.view(-1, VOCAB).float(), y.view(-1)).item())
    student.train()
    return sum(losses) / len(losses)

MICRO = 8
NMICRO = B // MICRO
t0 = time.time()
for step in range(start_step, STEPS):
    xs, ys = get_batch(train, B, T)
    opt.zero_grad()
    ce_sum, kd_sum = 0.0, 0.0
    for mi in range(NMICRO):
        xm, ym = xs[mi * MICRO:(mi + 1) * MICRO], ys[mi * MICRO:(mi + 1) * MICRO]
        if MODE.startswith("distill"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tl = teacher_forward(xm).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sl = student(xm)
        sl = sl.float()
        ce = F.cross_entropy(sl.view(-1, VOCAB), ym.view(-1))
        if MODE.startswith("distill"):
            T_ = TEMP
            sl2 = sl.view(-1, VOCAB)
            kd = F.kl_div(F.log_softmax(sl2 / T_, dim=-1),
                          F.softmax(tl.view(-1, VOCAB) / T_, dim=-1),
                          reduction="batchmean") * (T_ * T_)
            loss = ALPHA * ce + (1 - ALPHA) * kd
            kd_sum += kd.item()
        else:
            loss = ce
        (loss / NMICRO).backward()
        ce_sum += ce.item()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step(); sched.step()
    ce_avg, kd_avg = ce_sum / NMICRO, kd_sum / NMICRO
    if step % 200 == 0 or step == STEPS - 1:
        log(f"step {step}: ce {ce_avg:.4f}" +
            (f" kd {kd_avg:.4f}" if MODE == 'distill' else "") +
            f" lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")
    if (step + 1) % EVAL_EVERY == 0 or step == STEPS - 1:
        vl = eval_val()
        log(f"== step {step+1}: val_ce {vl:.4f} (ppl {math.exp(min(vl,20)):.2f})")
        save_ckpt(step + 1)

torch.save(student.state_dict(), os.path.join(OUT, f"student_{MODE}_{ARCH}.pt"))
log(f"DONE {MODE} {ARCH} in {time.time()-t0:.0f}s")
