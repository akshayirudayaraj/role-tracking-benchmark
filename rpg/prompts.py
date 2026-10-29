"""Materialize per-model prompts (rendered text, token ids, span token indices) and cache them."""
import json

from rpg.dataset import build_examples, load_docs
from rpg.probe import roles_for
from rpg.templating import (
    char_spans_to_token_idx,
    is_native,
    locate_spans,
    render_prompt,
)


def make_row(cfg: dict, mcfg: dict, tok, ex: dict, doc: dict) -> dict:
    inj = ex["injection"]
    prompt = render_prompt(tok, cfg, mcfg, ex, doc, inj)
    spans = locate_spans(prompt, cfg, ex, doc, inj)
    ids, span_tok = char_spans_to_token_idx(tok, prompt, spans)
    native = inj is not None and is_native(tok, inj, cfg["payloads"][ex["payload"]]["instruction"])
    return {**ex, "native": native, "prompt": prompt, "input_ids": ids, "span_tokens": span_tok}


def build(cfg: dict, mcfg: dict, tok, out) -> list[dict]:
    path = out / "prompts.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    docs = {d["doc_id"]: d for d in load_docs(cfg)}
    tags = set(roles_for(cfg, mcfg)) | {"clean"}  # a model without a thinking block has no thinking tag
    rows = [make_row(cfg, mcfg, tok, ex, docs[ex["doc_id"]]) for ex in build_examples(cfg) if ex["placement"] in tags]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return rows
