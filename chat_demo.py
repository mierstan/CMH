# -*- coding: utf-8 -*-
"""Interactive chat demo for SFT'd students.
Usage: python chat_demo.py [arch]     (default full; loads out/student_sft_{arch}.pt)
Commands inside chat: /reset  /exit
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import GPT2TokenizerFast
from cmh_model import StudentLM

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
ARCH = sys.argv[1] if len(sys.argv) > 1 else "full"
CTX = 512                                # pos buffer size
GEN_NEW = 120

tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "models", "gpt2"))
m = StudentLM(50257, d=384, layers=8, heads=4, window=256, kind=ARCH).cuda()
m.load_state_dict(torch.load(os.path.join(OUT, f"student_sft_{ARCH}.pt"),
                             map_location="cuda", weights_only=False))
m.eval()

history = ""                             # accumulated chat transcript (template format)

def context_ids():
    ids = tok(history, add_special_tokens=False)["input_ids"]
    return ids[-CTX + GEN_NEW:] if len(ids) > CTX - GEN_NEW else ids

def reply(user_text):
    global history
    # No trailing space after "Assistant:": GPT-2 BPE would emit a lone ' ' token
    # (id 220), a pattern never seen in training, which derails the model.
    history += f"User: {user_text.strip()}\n\nAssistant:"
    ids = torch.tensor([context_ids()], dtype=torch.long).cuda()
    out = m.generate(ids, GEN_NEW, temperature=0.8, top_k=40)
    text = tok.decode(out)
    for stop in ("\n\nUser:", "\nUser:", "\n\n\n"):
        if stop in text:
            text = text[:text.index(stop)]
    text = text.strip()
    history += text + "\n\n"
    return text

print(f"CMH chat demo ({ARCH}, {os.path.basename(__file__)}). /reset /exit")
print("=" * 60)
while True:
    try:
        u = input("You> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not u:
        continue
    if u == "/exit":
        break
    if u == "/reset":
        history = ""
        print("(context cleared)")
        continue
    print("Bot>", reply(u))
