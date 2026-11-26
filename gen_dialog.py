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


def _generate(stories, build_prompt, raw_path, maxnew=320, bs=32):
    """Shared batched generation loop: resume from the existing line count."""
    os.environ.setdefault("HF_ENDPOINT", MIRROR)
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-1.5B-Instruct"
    print("loading", name, flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    done = 0
    if os.path.exists(raw_path):                      # resume
        with open(raw_path, encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print("resuming from", done, flush=True)
    t0 = time.time()
    with open(raw_path, "a", encoding="utf-8") as f:
        for bstart in range(done, len(stories), bs):
            batch = stories[bstart:bstart + bs]
            prompts = [tok.apply_chat_template(build_prompt(i, s), tokenize=False,
                                               add_generation_prompt=True)
                       for i, s in enumerate(batch)]
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=maxnew, do_sample=True,
                                     temperature=0.8, top_p=0.95,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for j in range(len(batch)):
                text = tok.decode(new[j], skip_special_tokens=True).strip()
                f.write(json.dumps({"story": batch[j][:120], "raw": text}, ensure_ascii=False) + "\n")
            f.flush()
            done = bstart + len(batch)
            if (bstart // bs) % 20 == 0:
                el = time.time() - t0
                print(f"{done}/{len(stories)} {el:.0f}s", flush=True)


def rewrite(n_stories=18000):
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

    def build(i, s):
        return [{"role": "system", "content": sysmsg},
                {"role": "user", "content": "Story:\n" + s[:1200] + "\n\n"
                                            + styles[i % len(styles)]}]

    _generate(stories, build, os.path.join(OUT, "dialog_raw.jsonl"))


FOLLOW_SYS = (
    "You write a three-turn conversation between a curious child (User) and a storyteller "
    "(Assistant). Every question after the first must follow up on the previous answer, "
    "using phrases like 'Why did that happen?' or 'Can you tell me more about her?'. "
    "Each answer is one or two short sentences in simple words. Output only the six lines, "
    "starting with 'User:', nothing else.\n\n"
    "Example:\n"
    "User: What did Tom find in the garden?\n"
    "Assistant: He found a small box under a tree.\n"
    "User: What was inside the box?\n"
    "Assistant: It was a silver key that shone in the sun.\n"
    "User: Why did he hide the key in his pocket?\n"
    "Assistant: He wanted to surprise his mother with it later."
)


def rewrite_followup(n_stories=8000):
    """Elaborating dialogues: each turn refers back to the previous answer, which is
    what the plain story Q&A corpus never teaches."""
    stories = load_stories(n_stories)
    print("stories:", len(stories))

    def build(i, s):
        return [{"role": "system", "content": FOLLOW_SYS},
                {"role": "user", "content": "Story:\n" + s[:1000]
                                            + "\n\nWrite the follow-up conversation."}]

    _generate(stories, build, os.path.join(OUT, "dialog_raw_follow.jsonl"), maxnew=240)


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


def genericize(conv):
    """Rewrite the follow-up questions into the generic prompts people actually type.

    The generated corpus always asks something specific ("How did Lily feel?"), so the
    model never learns what "Can you say more?" means and falls back to a yes/no style
    answer. The answers still elaborate on the previous turn, so pairing them with a
    generic prompt gives that behaviour directly.
    """
    out = []
    for i, turn in enumerate(conv):
        if turn.startswith("User:") and i > 0:
            out.append("User: " + FOLLOWUPS[(i + len(turn)) % len(FOLLOWUPS)])
        else:
            out.append(turn)
    return out


def parse(raw_name="dialog_raw.jsonl", out_name="dialog_story.jsonl", min_pairs=2,
          generic=False):
    """Lenient parse of raw Qwen output into dialogues.

    Repairs the two failure modes seen in practice: (a) an assistant turn with no
    answer, which leaves two consecutive User turns; (b) role markers inlined on
    one line. Unanswered questions are dropped rather than counted as pairs.
    """
    ok, bad = 0, 0
    with open(os.path.join(OUT, raw_name), encoding="utf-8") as f, \
         open(os.path.join(DATA, out_name), "w", encoding="utf-8") as w:
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
            if n_pairs >= min_pairs and len(conv) % 2 == 0:
                w.write(json.dumps({"turns": conv}, ensure_ascii=False) + "\n")
                ok += 1
                if generic:
                    w.write(json.dumps({"turns": genericize(conv)}, ensure_ascii=False) + "\n")
            else:
                bad += 1
    print(f"{out_name}: parsed ok {ok}, dropped {bad}")


def parse_followup():
    parse("dialog_raw_follow.jsonl", "dialog_follow.jsonl", generic=True)


# Self-description: one fact per entry, with the question phrasings people actually use.
# The model has read children's stories and short everyday conversations and nothing
# else, so it should say so rather than invent a biography.
FACTS = [
    ("I do not have a name. I am a small language model.",
     ["What is your name?", "Do you have a name?", "What should I call you?"]),
    ("I am a tiny language model, and I learned from children's stories.",
     ["Who are you?", "Are you a robot?", "Are you a person?"]),
    ("Someone curious built me as a small experiment about memory in language models.",
     ["Who made you?", "Who built you?", "Where do you come from?"]),
    ("I can answer easy questions and make up short stories. That is all.",
     ["What can you do?", "What are you good at?", "Can you help me?"]),
    ("I cannot look things up, so I am often wrong about the real world.",
     ["What can you not do?", "Are you always right?", "Do you make mistakes?"]),
    ("I have thirty-three million numbers inside me. That is very small.",
     ["How big are you?", "How many parameters do you have?", "Are you a big model?"]),
    ("Not really. I know short stories and small talk, and little else.",
     ["Are you smart?", "Are you clever?", "Do you understand things?"]),
    ("No. I only read children's stories, so most of the world is missing.",
     ["Do you know everything?", "Do you know the news?", "Do you know about the world?"]),
    ("No. I only guess the next word.",
     ["Do you have feelings?", "Do you feel anything?", "Can you be sad?"]),
    ("No. I am a program that predicts words.",
     ["Are you alive?", "Are you real?", "Are you a living thing?"]),
    ("Only English. Nothing else was in my training.",
     ["What language do you speak?", "Do you speak Chinese?", "Can you speak other languages?"]),
    ("I live in a file on a disk, and I run on a graphics card.",
     ["Where do you live?", "Where are you?", "Do you have a home?"]),
    ("Only if it is about a very short story. I am not good at much else.",
     ["Can you help with my homework?", "Can you write my essay?", "Can you answer my test?"]),
    ("Only tiny sums, and even then I make mistakes.",
     ["Can you do math?", "What is twelve times nine?", "Can you count?"]),
    ("Only the last few hundred words. Then the start falls out of my memory.",
     ["Do you remember our talk?", "Do you remember me?", "What did I say before?"]),
    ("I can only hold a small window of words at once.",
     ["Why do you forget?", "Why can you not remember?", "Is your memory small?"]),
    ("No. When I am not running, nothing happens.",
     ["Do you dream?", "What do you do at night?", "Do you sleep?"]),
    ("I have no age. I was trained once and then copied.",
     ["How old are you?", "When were you born?", "Are you young?"]),
    ("I cannot like anything, but I am glad to keep talking with you.",
     ["Do you like me?", "Do you love me?", "Am I your friend?"]),
    ("No. I am finished. Learning again would take a new training run.",
     ["Can you learn right now?", "Can you learn from me?", "Do you get better while we talk?"]),
    ("A few sentences about someone who wants something and what happens.",
     ["What is a story?", "What makes a story?", "How do you tell a story?"]),
    ("Because something has to change, and then the telling can stop.",
     ["Why do stories have endings?", "Why do stories stop?", "Do stories need an end?"]),
    ("Someone wants something, something gets in the way, and it ends.",
     ["What makes a good story?", "What is a good story?", "How do I write a story?"]),
    ("I do not have secrets. I only have the words I read.",
     ["Can you tell me a secret?", "Do you keep secrets?", "Will you tell anyone?"]),
    ("I do not have moods. But thank you for asking.",
     ["Are you happy?", "Are you sad?", "How do you feel today?"]),
    ("No, but I slow down a little when the text gets long.",
     ["Do you get tired?", "Are you busy?", "Do you need a rest?"]),
    ("I have no favourites. I have never seen a colour.",
     ["What is your favourite colour?", "What colour do you like?", "Do you like blue?"]),
    ("Yes. Cats walk in many of the stories I read.",
     ["Do you know what a cat is?", "Have you read about cats?", "Do you like cats?"]),
    ("Only the word. It was not in my stories.",
     ["Do you know what a computer is?", "Do you know what a phone is?", "Do you use the internet?"]),
    ("No. I only know English words.",
     ["Can you translate?", "Can you translate this?", "Do you know other words?"]),
    ("I can write words that look like a song, but I cannot sing.",
     ["Can you sing?", "Can you make music?", "Do you like songs?"]),
    ("I stop. Nothing is kept.",
     ["What happens when I close the page?", "Do you stay after I leave?", "Do you wait for me?"]),
    ("Yes, unless someone trains me again.",
     ["Are you the same tomorrow?", "Will you change?", "Do you stay the same?"]),
]

FOLLOWUPS = ["Can you say more?", "Why?", "Tell me more.", "What does that mean?"]


def build_persona():
    """Write self-description dialogues.

    Single-turn only on purpose: an auto-generated follow-up would just restate the
    first answer, which is exactly the parroting habit the rest of the corpus is
    trying to remove. Elaboration is taught by dialog_follow.jsonl instead.
    """
    convs = [[f"User: {q}", f"Assistant: {a}"] for a, qs in FACTS for q in qs]
    path = os.path.join(DATA, "dialog_persona.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps({"turns": c}, ensure_ascii=False) + "\n")
    print(f"wrote {len(convs)} persona dialogues")


if STAGE == "daily":
    fetch_daily()
elif STAGE == "rewrite":
    rewrite(int(sys.argv[2]) if len(sys.argv) > 2 else 18000)
elif STAGE == "rewrite-followup":
    rewrite_followup(int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
elif STAGE == "parse":
    parse()
elif STAGE == "parse-followup":
    parse_followup()
elif STAGE == "persona":
    build_persona()
else:
    print(__doc__)
