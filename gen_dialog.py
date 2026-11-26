# -*- coding: utf-8 -*-
"""Dialogue SFT data generation.
Stages:
  python gen_dialog.py daily           -> data/dialog_daily.jsonl  (DailyDialog, hf-mirror)
  python gen_dialog.py rewrite [N]     -> out/dialog_raw.jsonl      (Qwen2.5-1.5B-Instruct rewrites
                                                                     TinyStories, streaming, resumable)
  python gen_dialog.py parse           -> data/dialog_story.jsonl   (lenient parse of raw output)
Conversation jsonl format: {"turns": ["User: ...", "Assistant: ...", ...]}
"""
import os, sys, json, re, time

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
STAGE = sys.argv[1] if len(sys.argv) > 1 else "daily"

MIRROR = "https://hf-mirror.com"


def fetch_daily():
    """Download DailyDialog (roskoN/dailydialog zips) through hf-mirror, parse to jsonl."""
    import urllib.request, zipfile, io
    urls = [
        f"{MIRROR}/datasets/roskoN/dailydialog/resolve/main/train.zip",
        f"{MIRROR}/datasets/roskoN/dailydialog/resolve/main/validation.zip",
    ]
    dialogs = []
    for url in urls:
        print("fetch", url)
        with urllib.request.urlopen(url, timeout=300) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
        names = [n for n in zf.namelist()
                 if n.endswith(".txt") and "act" not in n and "emotion" not in n]
        print("zip contents:", zf.namelist())
        text = zf.read(names[0]).decode("utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            turns = [t.strip().replace("__eou__", "").strip() for t in line.split("__eou__")]
            turns = [t for t in turns if t]
            if len(turns) >= 4:
                turns = turns[:8]
                conv = []
                for i, t in enumerate(turns):
                    conv.append(("User: " if i % 2 == 0 else "Assistant: ") + t)
                dialogs.append({"turns": conv})
        print("total so far", len(dialogs))
    with open(os.path.join(DATA, "dialog_daily.jsonl"), "w", encoding="utf-8") as f:
        for d in dialogs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("saved", len(dialogs), "dialogues")


def load_stories(n):
    path = os.path.join(DATA, "TinyStories-train.txt")
    stories = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        buf = []
        for line in f:
            if "<|endoftext|>" in line:
                s = " ".join(buf).strip()
                buf = []
                if 200 <= len(s) <= 1500:
                    stories.append(s)
                if len(stories) >= n:
                    break
            else:
                buf.append(line.strip())
    return stories


def rewrite(n_stories=18000):
    os.environ.setdefault("HF_ENDPOINT", MIRROR)
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-1.5B-Instruct"
    print("loading", name)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    stories = load_stories(n_stories)
    print("stories:", len(stories))
    sysmsg = ("You convert a children's story into a short dialogue between a curious child "
              "(User) and a friendly storyteller (Assistant). Exactly 3 exchanges: three lines "
              "starting with 'User:' each followed by one line starting with 'Assistant:'. "
              "The child asks about the story; the storyteller answers briefly in simple words. "
              "Output only the six lines, nothing else.")
    styles = [
        "Ask about what happened in the story.",
        "Ask about how the characters felt and why.",
        "Ask about a lesson the story teaches.",
    ]
    raw_path = os.path.join(OUT, "dialog_raw.jsonl")
    done = 0
    if os.path.exists(raw_path):                      # resume
        with open(raw_path, encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print("resuming from", done)
    BS, MAXNEW = 32, 320
    t0 = time.time()
    with open(raw_path, "a", encoding="utf-8") as f:
        for bstart in range(done, len(stories), BS):
            batch = stories[bstart:bstart + BS]
            prompts = []
            for i, s in enumerate(batch):
                style = styles[(bstart + i) % len(styles)]
                msgs = [{"role": "system", "content": sysmsg},
                        {"role": "user", "content": "Story:\n" + s[:1200] + "\n\n" + style}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=True,
                                     temperature=0.8, top_p=0.95,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for j in range(len(batch)):
                text = tok.decode(new[j], skip_special_tokens=True).strip()
                f.write(json.dumps({"story": batch[j][:120], "raw": text}, ensure_ascii=False) + "\n")
            f.flush()
            done = bstart + len(batch)
            if (bstart // BS) % 20 == 0:
                el = time.time() - t0
                print(f"{done}/{len(stories)} {el:.0f}s eta {el / max(done - bstart, 1) * (len(stories) - done):.0f}s")
    print("DONE rewrite")


def split_turns(raw):
    """Qwen sometimes emits 'Assistant: User: ...' on one line (answer missing),
    and sometimes glues the next turn onto the same line. Split on every role
    marker, not just the ones at line starts."""
    out = []
    for line in raw.splitlines():
        for part in re.split(r"\s*(?=(?:User|Assistant)\s*:)", line):
            m = re.match(r"\s*(User|Assistant)\s*:\s*(.*)$", part)
            if m:
                out.append((m.group(1), m.group(2).strip().strip('"').strip()))
    return out


def parse():
    """Lenient parse of raw Qwen output into dialogues.

    Repairs the two failure modes seen in practice: (a) an assistant turn with no
    answer, which leaves two consecutive User turns; (b) role markers inlined on
    one line. Unanswered questions are dropped rather than counted as pairs.
    """
    ok, bad = 0, 0
    with open(os.path.join(OUT, "dialog_raw.jsonl"), encoding="utf-8") as f, \
         open(os.path.join(DATA, "dialog_story.jsonl"), "w", encoding="utf-8") as w:
        for line in f:
            rec = json.loads(line)
            conv, expect = [], "User"
            for who, txt in split_turns(rec["raw"]):
                if not txt or len(txt) > 400:
                    continue
                if who == expect:
                    conv.append(f"{who}: {txt}")
                    expect = "Assistant" if who == "User" else "User"
                elif who == "User":
                    if conv and conv[-1].startswith("User"):   # previous question unanswered
                        conv.pop()
                    conv.append(f"{who}: {txt}")
                    expect = "Assistant"
            n_pairs = sum(1 for t in conv if t.startswith("User"))
            if n_pairs >= 2 and len(conv) % 2 == 0:
                w.write(json.dumps({"turns": conv}, ensure_ascii=False) + "\n")
                ok += 1
            else:
                bad += 1
    print(f"parsed ok {ok}, dropped {bad}")


if STAGE == "daily":
    fetch_daily()
elif STAGE == "rewrite":
    rewrite(int(sys.argv[2]) if len(sys.argv) > 2 else 18000)
elif STAGE == "parse":
    parse()
else:
    print(__doc__)
