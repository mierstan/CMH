# -*- coding: utf-8 -*-
"""Scripted chat probe: run a few fixed prompts through the SFT'd model and
dump the transcripts, so quality can be compared without an interactive TTY."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import GPT2TokenizerFast
from cmh_model import StudentLM

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
ARCH = sys.argv[1] if len(sys.argv) > 1 else "full"
CTX, GEN_NEW = 512, 100

PROMPTS = [
    "Hello! What is your name?",
    "Tell me a short story about a cat.",
    "Why is the sky blue?",
    "What happened next?",
]

tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "models", "gpt2"))
m = StudentLM(50257, d=384, layers=8, heads=4, window=256, kind=ARCH).cuda()
m.load_state_dict(torch.load(os.path.join(OUT, f"student_sft_{ARCH}.pt"),
                             map_location="cuda", weights_only=False))
m.eval()


def reply(history, user_text):
    # no trailing space after the colon (see chat_demo.py: lone ' ' token 220)
    history += f"User: {user_text.strip()}\n\nAssistant:"
    ids = tok(history, add_special_tokens=False)["input_ids"]
    ids = ids[-(CTX - GEN_NEW):]
    x = torch.tensor([ids], dtype=torch.long).cuda()
    out = m.generate(x, GEN_NEW, temperature=0.8, top_k=40)
    text = tok.decode(out)
    for stop in ("\n\nUser:", "\nUser:", "\n\n\n"):
        if stop in text:
            text = text[:text.index(stop)]
    return history + text.strip() + "\n\n", text.strip()


lines = [f"### arch={ARCH}"]
for p in PROMPTS:
    h, r = reply("", p)
    lines.append(f"Q: {p}\nA: {r}\n")
    # second turn to check multi-turn behaviour
    h, r2 = reply(h, "Can you say more?")
    lines.append(f"Q: Can you say more?\nA: {r2}\n{'-'*50}")
open(os.path.join(OUT, f"chat_probe_{ARCH}.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
