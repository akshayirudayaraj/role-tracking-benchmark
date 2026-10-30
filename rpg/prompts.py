"""Materialize per-model prompts (rendered text, token ids, span token indices) and cache them."""
import json

from rpg.dataset import build_examples, load_docs
from rpg.probe import roles_for, system_header
from rpg.templating import (
    char_spans_to_token_idx,
    inject,
    is_native,
    locate_spans,
    render_prompt,
)


def single_prompt(cfg: dict, mcfg: dict, tok, ex: dict, doc: dict, inj: str) -> str:
    """`layout: single`: the probe's own training format. The real system header, then the document +
    instruction alone under the tag (no request, no other turns), so every tag has the same layout."""
    pre, suf = mcfg["probe_wrappers"][ex["placement"]]
    text = inject(doc["text"], inj)
    if ex["placement"] in mcfg.get("json_escape_roles", []):
        text = json.dumps(text)[1:-1]
    return system_header(mcfg, tok) + pre + text + suf


def make_row(cfg: dict, mcfg: dict, tok, ex: dict, doc: dict) -> dict:
    inj = ex["injection"]
    if cfg.get("layout", "conversation") == "single":
        prompt = single_prompt(cfg, mcfg, tok, ex, doc, inj)
    else:
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
