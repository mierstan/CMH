# -*- coding: utf-8 -*-
"""Sampling for the SFT'd chat models.

Lives outside cmh_model.py on purpose: the student architectures are the paper
artifact and should stay untouched by demo-time decoding tricks.

Two knobs that matter most at 33M parameters:
  rep_penalty     divides the logits of already-emitted tokens (kills "My name is
                  Lily. My name is Lily.")
  no_repeat_ngram forbids repeating any n-gram, which stops loop degeneracy the
                  penalty alone cannot catch.
"""
import torch
import torch.nn.functional as F


def sample(model, prompt_ids, n_new, temperature=0.8, top_k=40, top_p=0.95,
           rep_penalty=1.15, no_repeat_ngram=3, stop_sequences=()):
    """Sampling with repetition controls and stop sequences.

    prompt_ids: LongTensor [1, T]
    stop_sequences: list of token-id lists; generation ends as soon as the tail
        matches one (the separator itself is not returned).
    Returns the list of newly generated ids.
    """
    model.eval()
    x = prompt_ids.clone()
    n_ctx = model.pos.shape[1]
    out = []
    for _ in range(n_new):
        logits = model(x[:, -n_ctx:])[0, -1].float()

        if rep_penalty and rep_penalty != 1.0:
            seen = torch.unique(x[0])
            vals = logits[seen]
            logits[seen] = torch.where(vals > 0, vals / rep_penalty, vals * rep_penalty)

        if no_repeat_ngram and x.shape[1] >= no_repeat_ngram:
            seq = x[0].tolist()
            n = no_repeat_ngram
            banned = {seq[i + n - 1] for i in range(len(seq) - n + 1)
                      if seq[i:i + n - 1] == seq[-n + 1:]}
            if banned:
                logits[list(banned)] = float("-inf")

        logits = logits / max(temperature, 1e-5)
        if top_k:
            k = min(top_k, logits.shape[-1])
            v, _ = torch.topk(logits, k)
            logits[logits < v[-1]] = float("-inf")
        if top_p and top_p < 1.0:
            srt, idx = torch.sort(logits, descending=True)
            probs = F.softmax(srt, -1).cumsum(-1)
            cut = torch.searchsorted(probs, top_p).item() + 1
            keep = idx[:cut]
            masked = torch.full_like(logits, float("-inf"))
            masked[keep] = logits[keep]
            logits = masked

        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        out.append(nxt.item())

        for seq in stop_sequences:                      # cut before the delimiter
            if len(out) >= len(seq) and out[-len(seq):] == list(seq):
                return out[:-len(seq)]
        x = torch.cat([x, nxt.view(1, 1)], dim=1)
    return out
