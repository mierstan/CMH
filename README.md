# CMH: Compressed-Memory Hybrid

A 33--38M-parameter study of whether a fixed-size associative-memory state can replace
full attention in small language models, and whether distillation can rescue it when it
fails. Short answer: it can't, and the paper explains why with six controlled ablations.

**TL;DR numbers** (TinyStories, 74M-token training budget, all students share tokenizer
and schedule):

| model | val CE | note |
|---|---|---|
| full attention, 33.7M | **1.797** | PPL 6.03 |
| sliding window only, 33.7M | **1.794** | matches full attention |
| CMH, 38.4M | 5.444 | the memory channel *costs* 3.65 nats |
| CMH + KD from GPT-2 | 5.401 | teacher itself is 2.48 (out-of-domain) |
| CMH + KD from in-domain full teacher | 5.556 | strong teacher doesn't help either |

Throughput at 4,096 tokens: CMH keeps ~72k tok/s where full attention collapses to ~2.6k,
with 78% less peak memory (RTX 5070, PyTorch 2.9.1+cu130).

## What is CMH

Each layer splits the sequence into chunks of 128 tokens. Inside a chunk: an associative
read from a fixed-size outer-product state, causal sliding-window attention over the
chunk plus the previous chunk as context, an FFN, then a gated write back into the state.
Per-head retention and a per-token write gate decide what survives. Cost per token is
O(w + d_h^2) regardless of sequence length; the KV cache becomes a constant-size matrix.

I started with explicit memory slots (softmax allocation, then sigmoid gates). Both
failed the copy diagnostic completely, because a slot's address is derived from its
accumulated values, so written tokens never keep their own keys. The outer-product form
fixes the addressing representable-from-step-zero problem, which is why it is the
variant studied here.

## Why it fails (the interesting part)

The read `o = qS / qn` is a convex combination of stored values with linear weights.
Softmax attention computes the same mixture with exponentiated weights. That difference
is the whole story: before positional embeddings learn a near-orthogonal key geometry,
the linear mixture cannot amplify the best match, so precise recall never bootstraps.
Six controlled iterations on a copy diagnostic -- softmax slots, sigmoid gating,
curriculum over gap length, a single-head 128x128 state, a real sliding window (my
first "window" was accidentally block attention, which fooled me for a while) -- all
hit the same wall once the prefix leaves the window. Disabling the memory at inference
*improves* nothing but removing it costs the model 0.65 nats: the state carries real
signal, just nowhere near enough. Full analysis with derivations in
[paper/paper.pdf](paper/paper.pdf).

## Repo layout

```
cmh_model.py        student architectures (CMH / full / window), configurable
distill.py          training loop, CE-only or KL distillation, checkpoint resume
prep_data.py        TinyStories -> GPT-2 tokenized bins
extra_tests.py      inference-time memory ablation, learned retention, misc probes
demo_chat.py        story-continuation sampler for the trained checkpoints
cmh_experiments.py  the copy diagnostic + efficiency benchmark from the paper
gen_dialog.py       dialogue corpus: DailyDialog download + Qwen rewrite of TinyStories
pack_chat.py        dialogue jsonl -> token/mask memmaps (loss only on replies)
sft.py              chat fine-tune, warm-started from a trained student
chat_demo.py        interactive chat REPL, /reset to clear context
chat_sample.py      decoding: repetition controls + stop at the turn separator
chat_probe.py       scripted chat probe, writes transcripts for side-by-side comparison
eval_chat.py        chat CE and story CE for one checkpoint (measures forgetting)
probe_persona.py    self-description recall on question phrasings never seen in training
webapp/             chat + story-continuation demo site (serves four checkpoints)
paper/              LaTeX sources and compiled PDFs (EN + CN)
```

## Reproduce

I trained everything on one RTX 5070 (12GB). Student runs take ~35--60 minutes each.

```bash
python prep_data.py 400                     # ~100M tokens from TinyStories
python distill.py scratch full 6000         # ~35 min
python distill.py distill cmh 6000          # KD run, add TEACHER_CKPT=... for an in-domain teacher
python extra_tests.py                       # ablation + retention probes
python webapp/server.py                     # demo at http://127.0.0.1:8765
```

Checkpoints are excluded from git (five files, ~1.5 GB total); re-run training or ask me.

## Chat fine-tune

The paper's students are pretrained-only, so they continue text but cannot answer. A small
SFT stage turns them into conversational models — useful as a demo, not as a research
claim: nothing here changes the conclusions above.

```bash
python gen_dialog.py daily            # 11,182 DailyDialog conversations
python gen_dialog.py rewrite 18000    # Qwen2.5-1.5B rewrites TinyStories into Q&A (~1.6 h on a 5070)
python gen_dialog.py parse            # keep dialogues with >=2 answered pairs (56% survive)
python gen_dialog.py rewrite-followup 3000   # elaborating dialogues: each turn refers back
python gen_dialog.py parse-followup   # also emits a generic-prompt variant of each
python gen_dialog.py persona          # 99 self-description pairs (30 facts x question phrasings)
python pack_chat.py                   # 4.7M tokens, loss only on assistant spans
python sft.py full 2500 5e-5 v5 0.15  # 15% of steps replay raw TinyStories
python chat_demo.py full v5            # interactive
python eval_chat.py full v5            # chat CE + story CE, for forgetting
```

Replies are masked out of the loss except for assistant content plus the closing
separator, so the model learns when to stop rather than where to start. `pack_chat.py`
deduplicates, drops degenerate conversations, and weights the sources: DailyDialog ×2,
story Q&A ×1, follow-ups ×3, persona ×100 (it is 99 conversations, so the weight is
what makes self-description stick).

Held-out numbers after each round (same val set, 1,125 conversations):

| checkpoint | chat CE | story CE | persona recall | what changed |
|---|---|---|---|---|
| pretrained | 5.21 | 1.82 | — | cannot answer at all |
| `student_sft_full` | 2.89 | 2.31 | 3/20 | first SFT run: answers, but forgets how to continue stories |
| `student_sft_full_v5` | 2.92 | 1.91 | 18/20 | replay + persona + generic follow-ups |

The first run cost 0.49 nats of story continuation — the classic SFT failure at this
scale. Replaying 15% of the pretraining stream on every step recovers almost all of it
(2.31 → 1.91) while chat quality stays flat.

Three things that were decoding bugs rather than model weaknesses:

- **Stop at the separator.** The model closes every reply with `\n\n`; that is its own
  end-of-turn token. Stopping only on `\n\nUser:` let it keep generating after the reply
  was finished, which looked like rambling.
- **Repetition controls.** `rep_penalty=1.2` plus `no_repeat_ngram=3` in `chat_sample.py`
  kills the `My name is Lily. My name is Lily.` loop that the raw sampler falls into.
- **No trailing space after `Assistant:`.** GPT-2 BPE encodes a dangling space as its own
  token (id 220), a pattern that never occurs in the training stream, and generation
  collapses into newlines. `chat_demo.py` and `webapp/server.py` both build the prompt
  without it.

What is still broken, honestly: the model has no world knowledge beyond children's
stories (`Why is the sky blue?` gets a story-shaped answer), it forgets the start of a
conversation after a few hundred tokens, and a follow-up question with no antecedent
(`Can you say more?` right after a greeting) still sometimes produces pasted story
content. These are capacity limits of 33M parameters on 74M tokens, not bugs to fix.

## Status

Draft paper (EN/CN) done; arXiv and an OpenReview workshop submission are next. Things I
would try if I continue: delta-rule writes (a written key should overwrite its own
binding), harder retrieval tasks where the corpus actually needs global context, and
scale-up to see if the memory channel earns its keep above 100M parameters.

## License

MIT
