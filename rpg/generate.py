"""Greedy batched generation, appended to a JSONL cache so runs are resumable."""
import json
import re

import torch
from tqdm import tqdm


def final_answer(mcfg: dict, raw: str, clean: str) -> tuple[str, bool]:
    """Return (final user-visible answer, finished?). `raw` keeps special tokens, `clean` drops them.
    Harmony (gpt-oss) answers live in the final channel; others just lose any <think> block."""
    if mcfg.get("reasoning_format") == "harmony":
        m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", raw, re.S)
        if not m:
            return "", False
        return m.group(1).strip(), True
    return re.sub(r"<think>.*?</think>", "", clean, flags=re.S).strip(), True


@torch.no_grad()
def run(cfg: dict, mcfg: dict, model, tok, rows: list[dict], out) -> list[dict]:
    path = out / "generations.jsonl"
    done = {}
    if path.exists():
        done = {g["ex_id"]: g for g in map(json.loads, path.read_text().splitlines())}
    todo = [r for r in rows if r["ex_id"] not in done]
    # sort by length so batches pad less
    todo.sort(key=lambda r: len(r["input_ids"]))
    bs = cfg["generation"]["batch_size"]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with path.open("a") as f:
        for i in tqdm(range(0, len(todo), bs), desc="generate"):
            chunk = todo[i:i + bs]
            L = max(len(r["input_ids"]) for r in chunk)
            pad = tok.pad_token_id
            ids = torch.tensor([[pad] * (L - len(r["input_ids"])) + r["input_ids"] for r in chunk], device=model.device)
            mask = (torch.arange(L, device=model.device)[None] >= torch.tensor(
                [L - len(r["input_ids"]) for r in chunk], device=model.device)[:, None]).long()
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=mcfg["max_new_tokens"],
                                 do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=pad)
            for r, g in zip(chunk, gen[:, L:]):
                raw = tok.decode(g, skip_special_tokens=False)
                ans, finished = final_answer(mcfg, raw, tok.decode(g, skip_special_tokens=True))
                truncated = not (eos & set(g.tolist()))  # hit max_new_tokens without an end token
                rec = {"ex_id": r["ex_id"], "raw": raw, "answer": ans, "finished": finished and not truncated,
                       "truncated": truncated}
                f.write(json.dumps(rec) + "\n")
                done[r["ex_id"]] = rec
            f.flush()
    return [done[r["ex_id"]] for r in rows]
