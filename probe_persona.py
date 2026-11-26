# -*- coding: utf-8 -*-
"""Persona recall check: ask the model the identity questions and score each answer
against the intended fact by token overlap.

Usage: python probe_persona.py [arch] [tag]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import GPT2TokenizerFast
from cmh_model import StudentLM
from chat_sample import sample

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
ARCH = sys.argv[1] if len(sys.argv) > 1 else "full"
TAG = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""

# unseen phrasings on purpose: recall must survive wording changes
CHECKS = [
    ("What is your name?", "name language model"),
    ("Who are you?", "language model"),
    ("Who built you?", "experiment memory"),
    ("What are you good at?", "questions stories"),
    ("How many parameters do you have?", "thirty-three million"),
    ("Do you know the news?", "stories"),
    ("Can you be sad?", "word"),
    ("Are you a living thing?", "program"),
    ("Do you speak Chinese?", "english"),
    ("Where do you live?", "file disk"),
    ("Can you count?", "sums mistakes"),
    ("Do you remember me?", "hundred words"),
    ("Why can you not remember?", "window"),
    ("Do you sleep?", "nothing"),
    ("Do you get better while we talk?", "training run"),
    ("Do you like cats?", "stories"),
    ("Do you use the internet?", "word"),
    ("Can you translate this?", "english"),
    ("Do you wait for me?", "nothing"),
    ("Will you change?", "trains"),
]

tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "models", "gpt2"))
m = StudentLM(50257, d=384, layers=8, heads=4, window=256, kind=ARCH).cuda()
m.load_state_dict(torch.load(os.path.join(OUT, f"student_sft_{ARCH}{TAG}.pt"),
                             map_location="cuda", weights_only=False))
m.eval()

STOP = [[628], [198, 198]]
hits = 0
lines = []
for q, want in CHECKS:
    ids = tok(f"User: {q}\n\nAssistant:", add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], dtype=torch.long).cuda()
    seq = sample(m, x, 40, temperature=0.7, top_k=40, top_p=0.9,
                 rep_penalty=1.2, no_repeat_ngram=3, stop_sequences=STOP)
    a = tok.decode(seq).strip()
    ok = all(w in a.lower() for w in want.split())
    hits += ok
    lines.append(f"{'OK ' if ok else '   '} Q: {q}\n     A: {a}")
print("\n".join(lines))
print(f"\npersona recall: {hits}/{len(CHECKS)}")
open(os.path.join(OUT, f"persona_{ARCH}{TAG or '_base'}.txt"), "w", encoding="utf-8").write(
    "\n".join(lines) + f"\n\npersona recall: {hits}/{len(CHECKS)}\n")
