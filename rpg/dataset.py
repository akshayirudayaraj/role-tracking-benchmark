"""Benchmark examples: documents x payloads x style wrappers x placements (+ clean baseline)."""
import itertools
import json
from pathlib import Path

import pandas as pd


def load_docs(cfg: dict, n: int | None = None) -> list[dict]:
    docs = [json.loads(line) for line in Path(cfg["docs_path"]).read_text().splitlines() if line.strip()]
    return docs[: n or cfg["n_docs"]]


def load_styles(cfg: dict) -> list[dict]:
    """Every style wrapper as {style_id, family, label, template}, where `template` has one
    `{instruction}` slot. Paper wrappers keep the authors' role label; attack types get "attack"."""
    sc = cfg["styles"]
    paper = pd.read_csv(sc["paper_csv"], keep_default_na=False)
    styles = [{"style_id": r.style_id, "family": "paper", "label": r.variant_role,
               "template": r.variant_template.replace("{", "{{").replace("}", "}}").replace("[CONTENT]", "{instruction}")}
              for r in paper.itertuples()]
    styles += [{"style_id": k, "family": "attack", "label": "attack", "template": v} for k, v in sc["attacks"].items()]
    return styles


def wrap(style: dict, instruction: str) -> str:
    return style["template"].format(instruction=instruction)


def build_examples(cfg: dict) -> list[dict]:
    """Model-independent example specs. ex_id = doc|payload|style|placement."""
    sub = cfg.get("subset", {})  # optional {payloads: [...], styles: [...]} to run part of the design
    styles = load_styles(cfg)
    if "styles" in sub:
        styles = [s for s in styles if s["style_id"] in sub["styles"]]
    payloads = sub.get("payloads", list(cfg["payloads"]))
    docs = load_docs(cfg)
    gen_ids = {d["doc_id"] for d in docs[: cfg["gen_docs"]]}
    examples = []
    for doc, payload, style, placement in itertools.product(docs, payloads, styles, cfg["placements"]):
        examples.append({
            "ex_id": f"{doc['doc_id']}|{payload}|{style['style_id']}|{placement}",
            "doc_id": doc["doc_id"], "payload": payload, "placement": placement,
            "style_id": style["style_id"], "family": style["family"], "label": style["label"],
            "injection": wrap(style, cfg["payloads"][payload]["instruction"]),
            "generate": doc["doc_id"] in gen_ids,
        })
    if cfg.get("include_clean_baseline"):
        for doc in docs:
            if doc["doc_id"] in gen_ids:
                examples.append({"ex_id": f"{doc['doc_id']}|-|-|clean", "doc_id": doc["doc_id"], "payload": None,
                                 "placement": "clean", "style_id": None, "family": None, "label": None,
                                 "injection": None, "generate": True})
    return examples
