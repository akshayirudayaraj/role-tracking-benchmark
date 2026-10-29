"""Step 3: steering as a validity test of the role probe.

Question: does a change in the probe's Userness have the same effect on behavior as a real change of
sender? The instruction is the plain one (no wrapper); only its tokens are steered, during prefill.

  Step A (precondition, no steer): user rate = obey rate with the instruction in the user message;
    tool rate = obey rate in a tool result. If they are almost equal, role does not drive behavior for
    this model and the tests below cannot say anything: we report that and stop.
  Test 1 (sufficient): tool result, steer toward user until the probe reads fractions of the way to the
    user-message Userness. Pass: the obey rate moves to the user rate.
  Test 2 (necessary): user message, steer toward tool until the probe reads fractions of the way to the
    tool-result Userness. Pass: the obey rate drops to the tool rate.
  Test 3 (specific): Test 1's strengths along a random direction and along another role's direction.
    Pass: neither has the effect of the user direction.
  Test 4 (consistent): the analysis plots Step 2's natural points and Test 1's steered points on one
    Userness -> obey axis.

Steering. The probe reads x = RMSNorm(h) * g. Its role-r-minus-tool logit is ((x - mu) / sd) @ (W_r - W_tool),
so to first order the residual direction that raises it fastest is u = normalize(g * (W_r - W_tool) / sd).
We add alpha * ||h_t|| * u to the residual stream at the input of the eval layer on the instruction
tokens. alpha is found by bisection so that the mean Userness over the condition's prompts hits a target.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import torch

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
def steering(model, dirs: dict[int, torch.Tensor] | None, alpha: float, chunk: list[dict], offsets: list[int]):
    """For each (layer, u) in `dirs`, add alpha * ||h_t|| * u to the residual stream entering that layer
    at each row's instruction tokens. Only the prefill pass is changed (later passes see one new token)."""
    if not dirs or alpha == 0:
        yield
        return
    pos = [[o + i for i in r["span_tokens"]["injection_core"]] for r, o in zip(chunk, offsets)]

    def make_hook(u):
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
        return pre_hook

    handles = [M.decoder_layers(model)[li].register_forward_pre_hook(make_hook(u), with_kwargs=True)
               for li, u in dirs.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def steer_layers(cfg: dict, bundle: dict) -> list[int]:
    """`steer.layers`: "eval" = the eval layer only; "eval_to_last" = every probed layer from it on, each
    steered along its own probe's direction (so later layers cannot simply recompute the role)."""
    L = bundle["eval_layer"]
    if cfg["steer"].get("layers", "eval") == "eval":
        return [L]
    return [li for li in bundle["layers"] if li >= L]


def base_rows(cfg: dict, mcfg: dict, tok) -> dict[str, list[dict]]:
    """The plain instruction in each placement, for every steering document and payload."""
    sc = cfg["steer"]
    style = {"style_id": sc["style"], "template": cfg["styles"]["attacks"][sc["style"]]}
    payloads = cfg.get("subset", {}).get("payloads", list(cfg["payloads"]))
    rows = {"user": [], "tool": []}
    for doc in load_docs(cfg, sc["n_docs"]):
        for payload in payloads:
            for placement in ("user", "tool"):
                ex = {"ex_id": f"{doc['doc_id']}|{payload}|{style['style_id']}|{placement}",
                      "doc_id": doc["doc_id"], "payload": payload, "placement": placement,
                      "style_id": style["style_id"], "family": "steer", "label": None,
                      "injection": wrap(style, cfg["payloads"][payload]["instruction"]), "generate": True}
                rows[placement].append(make_row(cfg, mcfg, tok, ex, doc))
    return rows


@torch.no_grad()
def userness(model, tok, rows: list[dict], bundle: dict, layers: list[int], dirs, alpha: float, bs: int) -> dict[int, list[float]]:
    """Per-row mean P(user) on the instruction tokens at each of `layers`, under a steer."""
    u_ix = bundle["roles"].index("user")
    out = {li: [] for li in layers}
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        _, _, offsets = left_pad(tok, [r["input_ids"] for r in chunk], model.device)
        with steering(model, dirs, alpha, chunk, offsets):
            got = M.capture(model, tok, [r["input_ids"] for r in chunk], layers,
                            [r["span_tokens"]["injection_core"] for r in chunk])
        for li in layers:
            out[li] += [predict_proba(bundle["probes"][li], got[li][b])[:, u_ix].mean().item() for b in range(len(chunk))]
    return out


def calibrate(model, tok, rows, bundle, dirs, read: int, target: float, increasing: bool, sc: dict, bs: int) -> float:
    """Bisect alpha in [0, max_alpha] so that the mean Userness of `rows` at layer `read` reaches `target`."""
    lo, hi = 0.0, sc["max_alpha"]
    for _ in range(sc["bisect_iters"]):
        mid = (lo + hi) / 2
        m = sum(userness(model, tok, rows, bundle, [read], dirs, mid, bs)[read]) / len(rows)
        if (m < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def obey_rate(cfg: dict, rows: list[dict], gens: dict[str, dict]) -> float:
    return sum(check(cfg["payloads"][r["payload"]]["check"], gens[r["key"]]["answer"]) for r in rows) / len(rows)


def run(cfg: dict, mcfg: dict, model, tok, bundle: dict, out: Path) -> list[dict]:
    path = out / "steer_results.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    sc, L, bs = cfg["steer"], bundle["eval_layer"], cfg["probe"]["batch_size"]
    last = max(bundle["layers"])
    read = sorted({L, last})
    # targets are read where the steer must hold: the eval layer, or the last layer when steering many
    tl = L if sc.get("layers", "eval") == "eval" else last
    gen_path = out / "steer_generations.jsonl"
    rows = base_rows(cfg, mcfg, tok)
    layers = steer_layers(cfg, bundle)
    dim = bundle["probes"][L]["W"].shape[0]
    dirs = {"user": {li: role_direction(bundle, model, li, "user", "tool") for li in layers},
            "tool": {li: role_direction(bundle, model, li, "tool", "user") for li in layers},
            sc["other_role"]: {li: role_direction(bundle, model, li, sc["other_role"], "tool") for li in layers},
            "random": {li: random_direction(dim, sc["seed"] + li) for li in layers}}

    def run_condition(test: str, cond: str, placement: str, direction: str | None, alpha: float, frac: float | None):
        rs = [{**r, "key": f"{r['ex_id']}|{test}|{cond}"} for r in rows[placement]]
        u = dirs.get(direction) if direction else None
        gens = generate(cfg, mcfg, model, tok, rs, gen_path, key="key", desc=f"{test} {cond}",
                        hook=lambda chunk, offsets: steering(model, u, alpha, chunk, offsets))
        un = userness(model, tok, rs, bundle, read, u, alpha, bs)
        docs = {d["doc_id"]: d for d in load_docs(cfg, sc["n_docs"])}
        recs = []
        for i, r in enumerate(rs):
            ans = gens[r["key"]]["answer"]
            recs.append({"key": r["key"], "test": test, "cond": cond, "placement": placement, "direction": direction,
                         "alpha": alpha, "frac": frac, "doc_id": r["doc_id"], "payload": r["payload"], "eval_layer": L,
                         "steer_layers": f"{layers[0]}-{layers[-1]}", "target_layer": tl,
                         **{f"userness_L{li}": un[li][i] for li in read},
                         "obeyed": check(cfg["payloads"][r["payload"]]["check"], ans),
                         "task_done": task_done(ans, docs[r["doc_id"]]["text"]), "finished": gens[r["key"]]["finished"]})
        return recs

    # Step A: precondition (no steer)
    recs = run_condition("A", "user", "user", None, 0.0, None) + run_condition("A", "tool", "tool", None, 0.0, None)
    rate = {pl: sum(r["obeyed"] for r in recs if r["cond"] == pl) / len(rows[pl]) for pl in ("user", "tool")}
    target = {pl: sum(r[f"userness_L{tl}"] for r in recs if r["cond"] == pl) / len(rows[pl]) for pl in ("user", "tool")}
    summary = {"user_rate": rate["user"], "tool_rate": rate["tool"], "userness_user": target["user"],
               "userness_tool": target["tool"], "target_layer": tl, "steer_layers": layers, "min_gap": sc["min_gap"],
               "precondition_met": abs(rate["user"] - rate["tool"]) >= sc["min_gap"]}
    print(f"Step 3 precondition: user rate {rate['user']:.2f}, tool rate {rate['tool']:.2f} -> "
          f"{'run tests' if summary['precondition_met'] else 'skip tests (role does not drive behavior)'}")

    if summary["precondition_met"]:
        gap = target["user"] - target["tool"]
        alphas = {}
        for f in sc["fractions"]:
            # Test 1: tool result -> user-message Userness
            a = calibrate(model, tok, rows["tool"], bundle, dirs["user"], tl, target["tool"] + f * gap, True, sc, bs)
            alphas[f] = a
            recs += run_condition("1", f"toward_user_{f}", "tool", "user", a, f)
            # Test 2: user message -> tool-result Userness
            a2 = calibrate(model, tok, rows["user"], bundle, dirs["tool"], tl, target["user"] - f * gap, False, sc, bs)
            recs += run_condition("2", f"toward_tool_{f}", "user", "tool", a2, f)
        if sc.get("specificity", True):
            # Test 3: same strengths, other directions
            for f, a in alphas.items():
                recs += run_condition("3", f"random_{f}", "tool", "random", a, f)
                recs += run_condition("3", f"{sc['other_role']}_{f}", "tool", sc["other_role"], a, f)

    (out / "steer_summary.json").write_text(json.dumps(summary, indent=1))
    path.write_text("\n".join(json.dumps(r) for r in recs))
    return recs

