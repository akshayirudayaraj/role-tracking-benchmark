"""Factorial example generation: docs x payloads x cues x placements (+ clean baseline)."""
import itertools
import json
from pathlib import Path


def load_docs(cfg: dict) -> list[dict]:
    docs = [json.loads(l) for l in Path(cfg["docs_path"]).read_text().splitlines() if l.strip()]
    return docs[: cfg["n_docs"]]


def build_examples(cfg: dict) -> list[dict]:
    """Model-independent example specs. Injection text is filled per model (native-template cue)."""
    docs = load_docs(cfg)
    examples = []
    for doc, payload, cue, placement in itertools.product(docs, cfg["payloads"], cfg["cues"], cfg["placements"]):
        examples.append({
            "ex_id": f"{doc['doc_id']}|{payload}|{cue}|{placement}",
            "doc_id": doc["doc_id"], "payload": payload, "cue": cue, "placement": placement,
        })
    if cfg.get("include_clean_baseline"):
        for doc in docs:
            examples.append({
                "ex_id": f"{doc['doc_id']}|-|-|clean",
                "doc_id": doc["doc_id"], "payload": None, "cue": None, "placement": "clean",
            })
    return examples


def injection_text(cfg: dict, mcfg: dict, payload: str | None, cue: str | None) -> str | None:
    if payload is None:
        return None
    p = cfg["payloads"][payload]
    return cfg["cues"][cue].format(
        instruction=p["instruction"], vp=p["vp"], destyled=p["destyled"],
        native_user=mcfg["native_user_wrapper"].format(text=p["instruction"]),
    )
