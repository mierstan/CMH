# -*- coding: utf-8 -*-
"""Scripted chat probe: fixed prompts, fixed seed, transcripts written to disk.

Usage:
  python chat_probe.py <arch>            # tuned sampling (repetition controls on)
  python chat_probe.py <arch> --plain    # pre-tuning sampling, for comparison
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import GPT2TokenizerFast
from cmh_model import StudentLM
from chat_sample import sample

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
ARCH = sys.argv[1] if len(sys.argv) > 1 else "full"
PLAIN = "--plain" in sys.argv
TAG = ("_" + next((a.split("=")[1] for a in sys.argv if a.startswith("--tag=")), "")) \
    if any(a.startswith("--tag=") for a in sys.argv) else ""
CTX, GEN_NEW = 512, 100

PROMPTS = [
    "Hello! What is your name?",
    "Tell me a short story about a cat.",
    "Why is the sky blue?",
    "What should I do if I feel sad?",
    "Do you like the rain?",
    "What happened next?",
]

# The model was trained to close every reply with "\n\n"; that separator is its
# own end-of-turn signal, so stop there instead of hoping for "\n\nUser:".
STOP = [] if PLAIN else [[628], [198, 198]]
KW = dict(temperature=0.8, top_k=40, top_p=0.95, rep_penalty=1.0, no_repeat_ngram=0)
if not PLAIN:
    KW.update(rep_penalty=1.2, no_repeat_ngram=3, top_p=0.9, stop_sequences=STOP)

tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "models", "gpt2"))
m = StudentLM(50257, d=384, layers=8, heads=4, window=256, kind=ARCH).cuda()
m.load_state_dict(torch.load(os.path.join(OUT, f"student_sft_{ARCH}{TAG}.pt"),
                             map_location="cuda", weights_only=False))
m.eval()


def reply(history, user_text):
    # no trailing space after the colon: GPT-2 BPE would emit a lone ' ' (id 220)
    history += f"User: {user_text.strip()}\n\nAssistant:"
    ids = tok(history, add_special_tokens=False)["input_ids"][-(CTX - GEN_NEW):]
    x = torch.tensor([ids], dtype=torch.long).cuda()
    torch.manual_seed(1234)
    out = sample(m, x, GEN_NEW, **KW)
    text = tok.decode(out)
    for stop in ("\n\nUser:", "\nUser:"):
        if stop in text:
            text = text[:text.index(stop)]
    return history + text.strip() + "\n\n", text.strip()


lines = [f"### arch={ARCH} sampling={'plain' if PLAIN else 'tuned'}"]
for p in PROMPTS:
    h, r = reply("", p)
    lines.append(f"Q: {p}\nA: {r}\n")
    h, r2 = reply(h, "Can you say more?")
    lines.append(f"Q: Can you say more?\nA: {r2}\n{'-'*50}")
name = "chat_probe_%s%s%s.txt" % (ARCH, TAG, "_plain" if PLAIN else "")
open(os.path.join(OUT, name), "w", encoding="utf-8").write("\n".join(lines))
print("wrote", name)
