# -*- coding: utf-8 -*-
"""Pack dialogue jsonl into token + mask memmaps for masked-loss SFT.

Format per conversation: "User: ...\n\nAssistant: ...\n\nUser: ...\n\nAssistant: ..."
Loss mask = 1 on assistant-turn tokens + the closing separator (the model learns to stop).

Sources are deduplicated, then weighted: DailyDialog and the follow-up dialogues carry
real conversational structure, the story Q&A carries domain vocabulary, and the small
persona set is upsampled so self-description answers actually stick.

Usage: python pack_chat.py  -> data/train_chat.bin/.mask + data/val_chat.bin/.mask
"""
import os, sys, json, random, re

import numpy as np
from transformers import GPT2TokenizerFast

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TOK_DIR = os.path.join(ROOT, "models", "gpt2")
SEP = "\n\n"
DAILY_COPIES = 2
PERSONA_COPIES = 100
FOLLOW_COPIES = 3
VAL_PER_SOURCE = 400

random.seed(0)

# (file, copies in train)
SOURCES = [
    ("dialog_daily.jsonl", DAILY_COPIES),
    ("dialog_story.jsonl", 1),
    ("dialog_follow.jsonl", FOLLOW_COPIES),
    ("dialog_persona.jsonl", PERSONA_COPIES),
]


def load_jsonl(path):
    if not os.path.exists(path):
        print("  missing (skipped):", os.path.basename(path))
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["turns"] for l in f]


def dedupe(convs):
    seen, out = set(), []
    for turns in convs:
        key = " ".join(t.lower() for t in turns)
        if key in seen:
            continue
        seen.add(key)
        out.append(turns)
    return out


def degenerate(turns):
    """Drop conversations that teach nothing: empty answers, answers that just echo
    the question, or a single word answer repeated across every turn."""
    answers = [t[len("Assistant: "):].strip() for t in turns if t.startswith("Assistant")]
    if not answers:
        return True
    if any(len(a) < 2 for a in answers):
        return True
    if len(set(answers)) == 1 and len(answers) > 1:
        return True
    return False


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
    train, val = [], []
    for name, copies in SOURCES:
        convs = load_jsonl(os.path.join(DATA, name))
        if not convs:
            continue
        raw_n = len(convs)
        convs = [c for c in dedupe(convs) if not degenerate(c)]
        random.shuffle(convs)
        n_val = min(VAL_PER_SOURCE, max(1, len(convs) // 10))
        val += convs[:n_val]
        train += convs[n_val:] * copies
        print(f"  {name:24s} {raw_n:6d} raw -> {len(convs):6d} kept, "
              f"x{copies} -> {len(convs) - n_val if copies == 1 else (len(convs) - n_val) * copies}")
    random.shuffle(train)
    pack(tok, val, "val_chat")
    pack(tok, train, "train_chat")


if __name__ == "__main__":
    main()
