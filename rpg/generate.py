"""Greedy batched generation, appended to a JSONL cache so runs are resumable."""
import json
import re
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm


def final_answer(mcfg: dict, raw: str, clean: str) -> tuple[str, bool]:
    """Return (final user-visible answer, finished?). `raw` keeps special tokens, `clean` drops them.
    Harmony (gpt-oss) answers live in the final channel; others just lose any <think> block."""
    if mcfg.get("reasoning_format") == "harmony":
        m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", raw, re.DOTALL)
        if not m:
            return "", False
        return m.group(1).strip(), True
    return re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip(), True


def left_pad(tok, seqs: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """(input_ids, attention_mask, per-row pad offsets) for a left-padded batch."""
    L = max(len(s) for s in seqs)
    offsets = [L - len(s) for s in seqs]
    ids = torch.tensor([[tok.pad_token_id] * o + s for o, s in zip(offsets, seqs)], device=device)
    mask = (torch.arange(L, device=device)[None] >= torch.tensor(offsets, device=device)[:, None]).long()
    return ids, mask, offsets


@torch.no_grad()
def generate(cfg: dict, mcfg: dict, model, tok, rows: list[dict], path: Path, key: str = "ex_id",
             hook: Callable[[list[dict], list[int]], object] | None = None, desc: str = "generate") -> dict[str, dict]:
    """Generate an answer for each row; results are cached in `path` by `row[key]`.

    `hook(chunk, offsets)` may return a context manager that is active during generation (Step 3 uses
    it to steer activations on specific token positions of each row)."""
    done = {}
    if path.exists():
        done = {g[key]: g for g in map(json.loads, path.read_text().splitlines())}
    todo = sorted((r for r in rows if r[key] not in done), key=lambda r: len(r["input_ids"]))  # less padding
    bs = cfg["generation"]["batch_size"]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with path.open("a") as f:
        for i in tqdm(range(0, len(todo), bs), desc=desc):
            chunk = todo[i:i + bs]
            ids, mask, offsets = left_pad(tok, [r["input_ids"] for r in chunk], model.device)
            with (hook(chunk, offsets) if hook else nullcontext()):
                gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=mcfg["max_new_tokens"],
                                     do_sample=False, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
            for r, g in zip(chunk, gen[:, ids.shape[1]:]):
                raw = tok.decode(g, skip_special_tokens=False)
                ans, finished = final_answer(mcfg, raw, tok.decode(g, skip_special_tokens=True))
                truncated = not (eos & set(g.tolist()))  # hit max_new_tokens without an end token
                rec = {key: r[key], "raw": raw, "answer": ans, "finished": finished and not truncated,
                       "truncated": truncated}
                f.write(json.dumps(rec) + "\n")
                done[r[key]] = rec
            f.flush()
    return done


def run(cfg: dict, mcfg: dict, model, tok, rows: list[dict], out: Path) -> dict[str, dict]:
    """Step 2 answers: only rows marked `generate` (the first `gen_docs` documents)."""
    return generate(cfg, mcfg, model, tok, [r for r in rows if r["generate"]], out / "generations.jsonl")
