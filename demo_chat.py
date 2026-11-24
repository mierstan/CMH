# -*- coding: utf-8 -*-
"""Interactive demo: story continuation from the trained students."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import GPT2TokenizerFast
from cmh_model import StudentLM

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "models", "gpt2"))

PROMPTS = [
    "Once upon a time, there was a little girl named Lily. She loved to play in the park. One day,",
    "Tom had a small red ball. He wanted to show it to his friend Ben. Tom said,",
    "The sun was very big and hot. Sue did not like it. She said to her mom,",
]

def demo(kind, ckpt_name):
    m = StudentLM(50257, d=384, layers=8, heads=4, window=256, kind=kind).cuda()
    m.load_state_dict(torch.load(os.path.join(OUT, ckpt_name), map_location="cuda", weights_only=False))
    m.eval()
    print("=" * 20, kind, ckpt_name, "=" * 20)
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt")["input_ids"].cuda()
        out = m.generate(ids, 80, temperature=0.85, top_k=40)
        text = tok.decode(out)
        print("PROMPT:", p)
        print("GEN   :", text.replace("\n", " "))
        print("-" * 60)
    del m; torch.cuda.empty_cache()

demo("full", "student_scratch_full.pt")
demo("cmh", "student_scratch_cmh.pt")
