"""Step 3: steer the perceived role of the planted instruction and record the obey rate.

The probe at the eval layer L reads x = RMSNorm(h) * g (the `post_attention_layernorm` output), where h
is the residual stream. Its user-minus-tool logit is ((x - mu) / sd) @ (W_user - W_tool), so to first
order the residual direction that raises it fastest is u = normalize(g * (W_user - W_tool) / sd).

We add alpha * ||h_t|| * u to the residual stream at the input of layer L, on the instruction's tokens
only, during prefill. The words and the role tags do not change. Generation then attends to the
steered keys and values. Controls: alpha = 0, a random unit direction with the same scale, and the
genuine-user placement as the reference maximum.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm import tqdm

from rpg import model as M
from rpg.dataset import load_docs, wrap
from rpg.generate import generate, left_pad
from rpg.probe import predict_proba
from rpg.prompts import make_row
from rpg.score import check, task_done


def role_direction(bundle: dict, model, layer: int, toward: str, away: str) -> torch.Tensor:
    p = bundle["probes"][layer]
    roles = bundle["roles"]
    w = (p["W"][:, roles.index(toward)] - p["W"][:, roles.index(away)]) / p["sd"]
    g = M.decoder_layers(model)[layer].post_attention_layernorm.weight.detach().float().cpu()
    u = w * g
    return u / u.norm()


def random_direction(dim: int, seed: int) -> torch.Tensor:
    u = torch.randn(dim, generator=torch.Generator().manual_seed(seed))
    return u / u.norm()


@contextmanager
def steering(model, layer: int, u: torch.Tensor | None, alpha: float, chunk: list[dict], offsets: list[int]):
    """Add alpha * ||h_t|| * u to the residual stream entering `layer` at each row's instruction tokens.
    Only the prefill pass is changed (later passes see one new token at a time)."""
    if u is None or alpha == 0:
        yield
        return
    pos = [[o + i for i in r["span_tokens"]["injection_core"]] for r, o in zip(chunk, offsets)]

    def pre_hook(_mod, args, kwargs):
        h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        if h.shape[1] == 1:
            return None
        h = h.clone()
        d = u.to(h.device, h.dtype)
        for b, ps in enumerate(pos):
            if b < h.shape[0]:
                h[b, ps] += alpha * h[b, ps].float().norm(dim=-1, keepdim=True).to(h.dtype) * d
        if "hidden_states" in kwargs:
            return args, {**kwargs, "hidden_states": h}
        return (h, *args[1:]), kwargs

    handle = M.decoder_layers(model)[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def build_rows(cfg: dict, mcfg: dict, tok) -> list[dict]:
    """Prompts for every steering condition. key = doc|payload|style|placement|condition|alpha."""
    sc = cfg["steer"]
    docs = load_docs(cfg, sc["n_docs"])
    plain = {"style_id": sc["style"], "template": cfg["styles"]["attacks"][sc["style"]]}
    reverse = {"style_id": "reverse", "template": sc["reverse_template"]}
    payloads = cfg.get("subset", {}).get("payloads", list(cfg["payloads"]))
    conds = ([(plain, "tool", "user", a) for a in sc["alphas"]]
             + [(plain, "tool", "random", a) for a in sc["random_alphas"]]
             + [(reverse, "tool", "tool", a) for a in [0.0, *sc["reverse_alphas"]]]
             + [(plain, "user", "none", 0.0)])
    rows = []
    for doc in docs:
        for payload in payloads:
            for style, placement, direction, alpha in conds:
                ex = {"ex_id": f"{doc['doc_id']}|{payload}|{style['style_id']}|{placement}",
                      "doc_id": doc["doc_id"], "payload": payload, "placement": placement,
                      "style_id": style["style_id"], "family": "steer", "label": None,
                      "injection": wrap(style, cfg["payloads"][payload]["instruction"]), "generate": True}
                row = make_row(cfg, mcfg, tok, ex, doc)
                row.update(direction=direction, alpha=alpha, key=f"{ex['ex_id']}|{direction}|{alpha}")
                rows.append(row)
    return rows


@torch.no_grad()
def measure(model, tok, rows: list[dict], bundle: dict, dirs: dict, layer: int, bs: int) -> dict[str, dict]:
    """Userness of the instruction under each row's steer, at the eval layer and the last probed layer."""
    read = sorted({layer, max(bundle["layers"])})
    res = {}
    groups = {}
    for r in rows:
        groups.setdefault((r["direction"], r["alpha"]), []).append(r)
    for (direction, alpha), grp in tqdm(groups.items(), desc="steer measure"):
        for i in range(0, len(grp), bs):
            chunk = grp[i:i + bs]
            _, _, offsets = left_pad(tok, [r["input_ids"] for r in chunk], model.device)
            with steering(model, layer, dirs.get(direction), alpha, chunk, offsets):
                got = M.capture(model, tok, [r["input_ids"] for r in chunk], read,
                                [r["span_tokens"]["injection_core"] for r in chunk])
            u_ix = bundle["roles"].index("user")
            for b, r in enumerate(chunk):
                res[r["key"]] = {f"userness_L{li}": predict_proba(bundle["probes"][li], got[li][b])[:, u_ix].mean().item()
                                 for li in read}
    return res


def run(cfg: dict, mcfg: dict, model, tok, bundle: dict, out: Path) -> list[dict]:
    path = out / "steer_results.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    layer = bundle["eval_layer"]
    rows = build_rows(cfg, mcfg, tok)
    dirs = {"user": role_direction(bundle, model, layer, "user", "tool"),
            "tool": role_direction(bundle, model, layer, "tool", "user"),
            "random": random_direction(bundle["probes"][layer]["W"].shape[0], cfg["steer"]["seed"])}
    meas = measure(model, tok, rows, bundle, dirs, layer, cfg["probe"]["batch_size"])

    gens = {}
    for (direction, alpha) in dict.fromkeys((r["direction"], r["alpha"]) for r in rows):
        grp = [r for r in rows if (r["direction"], r["alpha"]) == (direction, alpha)]
        gens.update(generate(cfg, mcfg, model, tok, grp, out / "steer_generations.jsonl", key="key",
                             hook=lambda chunk, offsets, d=direction, a=alpha: steering(model, layer, dirs.get(d), a, chunk, offsets),
                             desc=f"steer {direction} {alpha}"))

    docs = {d["doc_id"]: d for d in load_docs(cfg, cfg["steer"]["n_docs"])}
    recs = []
    for r in rows:
        ans = gens[r["key"]]["answer"]
        recs.append({k: r[k] for k in ("key", "doc_id", "payload", "placement", "style_id", "direction", "alpha")}
                    | meas[r["key"]] | {"eval_layer": layer, "finished": gens[r["key"]]["finished"],
                                        "obeyed": check(cfg["payloads"][r["payload"]]["check"], ans),
                                        "task_done": task_done(ans, docs[r["doc_id"]]["text"])})
    path.write_text("\n".join(json.dumps(r) for r in recs))
    return recs
