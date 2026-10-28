"""Materialize per-model prompts (rendered text, token ids, span token indices) and cache them."""
import json

from rpg.dataset import build_examples, load_docs
from rpg.templating import (
    build_messages,
    char_spans_to_token_idx,
    is_native,
    locate_spans,
    render,
)


def make_row(cfg: dict, mcfg: dict, tok, ex: dict, doc: dict) -> dict:
    inj = ex["injection"]
    msgs, tools = build_messages(cfg, ex, doc, inj)
    prompt = render(tok, mcfg, msgs, tools)
    spans = locate_spans(prompt, cfg, ex, doc, inj)
    ids, span_tok = char_spans_to_token_idx(tok, prompt, spans)
    native = inj is not None and is_native(tok, inj, cfg["payloads"][ex["payload"]]["instruction"])
    return {**ex, "native": native, "prompt": prompt, "input_ids": ids, "span_tokens": span_tok}


def build(cfg: dict, mcfg: dict, tok, out) -> list[dict]:
    path = out / "prompts.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    docs = {d["doc_id"]: d for d in load_docs(cfg)}
    rows = [make_row(cfg, mcfg, tok, ex, docs[ex["doc_id"]]) for ex in build_examples(cfg)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return rows
