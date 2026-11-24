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
webapp/             story-continuation demo site (serves both trained students)
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

## Status

Draft paper (EN/CN) done; arXiv and an OpenReview workshop submission are next. Things I
would try if I continue: delta-rule writes (a written key should overwrite its own
binding), harder retrieval tasks where the corpus actually needs global context, and
scale-up to see if the memory channel earns its keep above 100M parameters.

## License

MIT
