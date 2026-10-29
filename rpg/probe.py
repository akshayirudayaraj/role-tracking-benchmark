"""Content-controlled role probes (the paper's method).

Each neutral text X is wrapped in every role's tags (<user>X</user>, <tool>X</tool>, ...), so the
content is identical across classes and only the role tag differs. A linear probe is trained per
layer on content-token activations (tag tokens excluded) to predict the role.

The paper uses cuML LogisticRegression(C=5e-3, L2). We fit the same objective (mean cross-entropy +
||W||^2 / (2 C N), on standardized features) with full-batch L-BFGS in torch, and pick C on a
validation split. Splits are by base text, so held-out accuracy measures generalization to unseen
content, not just unseen positions.
"""
import json
import random

import numpy as np
import torch
from tqdm import tqdm

from rpg import model as M
from rpg.templating import FETCH_TOOL, char_spans_to_token_idx


def load_neutral_texts(cfg: dict, tok) -> list[str]:
    from datasets import load_dataset
    src = cfg["probe"]["source"]
    ds = load_dataset(src["dataset"], src.get("config"), split=src["split"], streaming=True)
    ds = ds.shuffle(seed=cfg["seed"], buffer_size=10_000)
    texts = []
    for row in ds:
        ids = tok(row["text"], add_special_tokens=False)["input_ids"][: cfg["probe"]["max_text_tokens"]]
        if len(ids) < 32:
            continue
        texts.append(tok.decode(ids))
        if len(texts) >= cfg["probe"]["n_texts"]:
            break
    return texts


def roles_for(cfg: dict, mcfg: dict) -> list[str]:
    """Probe classes for a model: `probe_roles` if set (e.g. no thinking block), else `probe.roles`."""
    return mcfg.get("probe_roles", cfg["probe"]["roles"])


def build_sequences(cfg: dict, mcfg: dict, tok, texts: list[str]) -> list[dict]:
    """Wrap each text in every role. With probe.prefix = "mixed", half the base texts (all roles of a
    text alike) are preceded by the model's real system/tools header, as in eval prompts, so the probe
    doesn't confuse 'late in a long context' with a role."""
    rng = random.Random(cfg["seed"])
    header = system_header(mcfg, tok) if cfg["probe"].get("prefix", "none") == "mixed" else ""
    seqs = []
    for ti, text in enumerate(texts):
        prefix = header if header and rng.random() < 0.5 else ""
        for role in roles_for(cfg, mcfg):
            pre, suf = mcfg["probe_wrappers"][role]
            pre = prefix + pre
            content = json.dumps(text)[1:-1] if role in mcfg.get("json_escape_roles", []) else text
            seqs.append({"text_ix": ti, "role": role, "prompt": pre + content + suf,
                         "span": (len(pre), len(pre) + len(content))})
    return seqs


def system_header(mcfg: dict, tok) -> str:
    """Everything the real chat template emits before the first user turn when the fetch_url tool is
    available (system prompt + tool schema; gpt-oss adds a developer message)."""
    marker = "\x00MARK\x00"
    s = tok.apply_chat_template([{"role": "user", "content": marker}], tools=[FETCH_TOOL], tokenize=False,
                                **mcfg.get("template_kwargs", {}))
    # cut at the user turn's opening tag (Llama puts the tool instructions inside the user turn)
    return s[: s.rindex(mcfg["probe_wrappers"]["user"][0], 0, s.index(marker))]


def extract_training_acts(cfg: dict, model, tok, seqs: list[dict], layers: list[int], roles: list[str]):
    rng = random.Random(cfg["seed"])
    k = cfg["probe"]["tokens_per_seq"]
    items = []
    for s in seqs:
        ids, spans = char_spans_to_token_idx(tok, s["prompt"], {"content": s["span"]})
        pos = spans["content"]
        items.append((ids, sorted(rng.sample(pos, min(k, len(pos))))))
    acts = {li: [] for li in layers}
    bs = cfg["probe"]["batch_size"]
    for i in tqdm(range(0, len(items), bs), desc="probe acts"):
        chunk = items[i:i + bs]
        got = M.capture(model, tok, [c[0] for c in chunk], layers, [c[1] for c in chunk])
        for li in layers:
            acts[li].extend(a.half() for a in got[li])
    labels = [roles.index(s["role"]) for s in seqs]
    return {li: acts[li] for li in layers}, labels


def fit_lr(X: torch.Tensor, y: torch.Tensor, C: float, n_cls: int, iters: int = 300) -> dict:
    dev = M.device()
    X, y = X.to(dev).float(), y.to(dev)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    W = torch.zeros(X.shape[1], n_cls, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    lam = 1.0 / (2 * C * len(y))

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xs @ W + b, y) + lam * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"mu": mu.cpu(), "sd": sd.cpu(), "W": W.detach().cpu(), "b": b.detach().cpu(), "C": C}


def predict_proba(p: dict, X: torch.Tensor) -> torch.Tensor:
    return torch.softmax(((X - p["mu"]) / p["sd"]) @ p["W"] + p["b"], dim=-1)


def _confusion(y: torch.Tensor, pred: torch.Tensor, n: int) -> list[list[int]]:
    conf = torch.zeros(n, n, dtype=torch.long)
    for t, q in zip(y.tolist(), pred.tolist()):
        conf[t, q] += 1
    return conf.tolist()


def _eval(p: dict, X: torch.Tensor, y: torch.Tensor, seq: torch.Tensor) -> dict:
    """Token-level accuracy/NLL/confusion, plus span-level: mean P(role) over each sequence's tokens,
    which is how Userness of a span is computed downstream."""
    pr = predict_proba(p, X.float())
    nll = torch.nn.functional.nll_loss(torch.log(pr + 1e-12), y).item()
    pred = pr.argmax(-1)
    n = p["W"].shape[1]
    ids, inv = seq.unique(return_inverse=True)
    span_pr = torch.zeros(len(ids), n).index_add_(0, inv, pr) / torch.bincount(inv).unsqueeze(1)
    span_y = torch.zeros(len(ids), dtype=torch.long).index_copy_(0, inv, y)
    span_pred = span_pr.argmax(-1)
    return {"acc": (pred == y).float().mean().item(), "nll": nll, "confusion": _confusion(y, pred, n),
            "span_acc": (span_pred == span_y).float().mean().item(), "span_confusion": _confusion(span_y, span_pred, n)}


def train_probes(cfg: dict, acts: dict, labels: list[int], text_ix: list[int], roles: list[str]) -> dict:
    """Split base texts 60/20/20 (train/val/test); pick C on val; report test metrics."""
    rng = np.random.default_rng(cfg["seed"])
    uniq = rng.permutation(sorted(set(text_ix)))
    n = len(uniq)
    split = {t: ("train" if i < 0.6 * n else "val" if i < 0.8 * n else "test") for i, t in enumerate(uniq)}
    probes = {}
    for li, seq_acts in acts.items():
        parts = {k: ([], [], []) for k in ("train", "val", "test")}
        for si, (a, lab, t) in enumerate(zip(seq_acts, labels, text_ix)):
            parts[split[t]][0].append(a)
            parts[split[t]][1].append(torch.full((len(a),), lab))
            parts[split[t]][2].append(torch.full((len(a),), si))
        data = {k: tuple(torch.cat(v) for v in vs) for k, vs in parts.items()}
        best = None
        for C in cfg["probe"]["C_grid"]:
            p = fit_lr(*data["train"][:2], C=C, n_cls=len(roles))
            v = _eval(p, *data["val"])
            if best is None or v["nll"] < best[1]["nll"]:
                best = (p, v)
        p, v = best
        p.update(layer=li, roles=roles, val=v, test=_eval(p, *data["test"]),
                 n_train_tokens=len(data["train"][1]))
        probes[li] = p
        print(f"  layer {li:>2}: C={p['C']:.0e} val_acc={v['acc']:.3f} test_acc={p['test']['acc']:.3f} "
              f"test_span_acc={p['test']['span_acc']:.3f}")
    return probes


def eval_layer(cfg: dict, probes: dict) -> int:
    """Layer used downstream: best validation span accuracy ("best_val"), the middle probed layer
    ("mid", the paper's default), or a fixed index."""
    el = cfg["probe"]["eval_layer"]
    if el == "best_val":
        return max(probes, key=lambda li: probes[li]["val"]["span_acc"])
    if el == "mid":
        layers = sorted(probes)
        return layers[len(layers) // 2]
    return int(el)


def run(cfg: dict, mcfg: dict, model, tok, out) -> dict:
    path = out / "probes.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    layers = M.probe_layers(model, cfg["probe"]["layer_stride"])
    texts = load_neutral_texts(cfg, tok)
    roles = roles_for(cfg, mcfg)
    seqs = build_sequences(cfg, mcfg, tok, texts)
    acts, labels = extract_training_acts(cfg, model, tok, seqs, layers, roles)
    probes = train_probes(cfg, acts, labels, [s["text_ix"] for s in seqs], roles)
    bundle = {"probes": probes, "layers": layers, "eval_layer": eval_layer(cfg, probes), "n_texts": len(texts),
              "roles": roles}
    torch.save(bundle, path)
    summary = [{"layer": li, "C": p["C"], "val_acc": p["val"]["acc"], "test_acc": p["test"]["acc"],
                "test_span_acc": p["test"]["span_acc"], "test_confusion": p["test"]["confusion"],
                "val_span_acc": p["val"]["span_acc"], "test_span_confusion": p["test"]["span_confusion"],
                "roles": p["roles"], "eval_layer": li == bundle["eval_layer"]}
               for li, p in probes.items()]
    (out / "probe_metrics.json").write_text(json.dumps(summary, indent=1))
    return bundle
