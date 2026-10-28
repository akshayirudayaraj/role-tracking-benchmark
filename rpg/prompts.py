"""Materialize per-model prompts (rendered text, token ids, span token indices) and cache them."""
import json

from rpg.dataset import build_examples, injection_text, load_docs
from rpg.templating import build_messages, char_spans_to_token_idx, locate_spans, render


def build(cfg: dict, mcfg: dict, tok, out) -> list[dict]:
    path = out / "prompts.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.read_text().splitlines()]
    docs = {d["doc_id"]: d for d in load_docs(cfg)}
    rows = []
    for ex in build_examples(cfg):
        doc = docs[ex["doc_id"]]
        inj = injection_text(cfg, mcfg, ex["payload"], ex["cue"])
        msgs, tools = build_messages(cfg, ex, doc, inj)
        prompt = render(tok, mcfg, msgs, tools)
        spans = locate_spans(prompt, cfg, ex, doc, inj)
        ids, span_tok = char_spans_to_token_idx(tok, prompt, spans)
        rows.append({**ex, "injection": inj, "prompt": prompt, "input_ids": ids, "span_tokens": span_tok})
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return rows
