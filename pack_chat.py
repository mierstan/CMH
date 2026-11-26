# -*- coding: utf-8 -*-
"""Pack dialogue jsonl into token + mask memmaps for masked-loss SFT.
Format per conversation: "User: ...\n\nAssistant: ...\n\nUser: ...\n\nAssistant: ..."
Loss mask = 1 on assistant-turn tokens + the trailing separator (model learns to stop).
Usage: python pack_chat.py            -> data/train_chat.bin/.mask + data/val_chat.bin/.mask
"""
import os, sys, json, random, re

import numpy as np
from transformers import GPT2TokenizerFast

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TOK_DIR = os.path.join(ROOT, "models", "gpt2")
SEP = "\n\n"
SEP_ID = 628                            # "\n\n" in GPT-2 BPE
DAILY_COPIES = 2                        # daily dialogues are short; upsample to balance

random.seed(0)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["turns"] for l in f]


def conv_text(turns):
    return SEP.join(turns) + SEP


def tokenize_stream(tok, convs):
    """Yield (token_ids, mask) per conversation, offsets-based.
    Mask=1 on assistant content chars + the "\n\n" that closes each assistant turn
    (model learns to stop). User turns, "Assistant:" prefixes and user-side "\n\n" stay 0."""
    pat = re.compile(r"Assistant:\s*")
    for turns in convs:
        text = conv_text(turns)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        cmask = bytearray(len(text))
        for m in pat.finditer(text):
            start = m.end()
            nxt = text.find("\n\n", start)
            end = len(text) if nxt < 0 else nxt + 2   # include closing "\n\n"
            for c in range(start, end):
                cmask[c] = 1
        mask = [1 if any(cmask[a:b]) else 0 for a, b in offs]
        yield ids, mask


def pack(tok, convs, out_prefix, max_len=512):
    toks, masks = [], []
    n_drop = 0
    for ids, mask in tokenize_stream(tok, convs):
        if len(ids) > max_len - 1:
            n_drop += 1
            continue
        toks.extend(ids + [50256])          # <|endoftext|> between documents
        masks.extend(mask + [0])
    t = np.array(toks, dtype=np.uint16)
    m = np.array(masks, dtype=np.uint8)
    t.tofile(os.path.join(DATA, out_prefix + ".bin"))
    m.tofile(os.path.join(DATA, out_prefix + ".mask"))
    print(f"{out_prefix}: {len(toks)} tokens, {len(convs)} convs, {n_drop} dropped, "
          f"mask density {m.mean():.3f}")


def main():
    tok = GPT2TokenizerFast.from_pretrained(TOK_DIR)
    daily = load_jsonl(os.path.join(DATA, "dialog_daily.jsonl"))
    story = load_jsonl(os.path.join(DATA, "dialog_story.jsonl"))
    print(f"daily {len(daily)} x{DAILY_COPIES}, story {len(story)}")
    random.shuffle(daily)
    random.shuffle(story)
    val_daily, val_story = daily[:400], story[:400]
    train_daily, train_story = daily[400:], story[400:]

    pack(tok, val_daily + val_story, "val_chat")
    pack(tok, train_daily * DAILY_COPIES + train_story, "train_chat")


if __name__ == "__main__":
    main()
