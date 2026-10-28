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


def build_sequences(cfg: dict, mcfg: dict, tok, texts: list[str]) -> list[dict]:
    """Wrap each text in every role. With probe.prefix = "mixed", half the base texts (all roles of a
    text alike) are preceded by the model's real system/tools header, as in eval prompts, so the probe
    doesn't confuse 'late in a long context' with a role."""
    rng = random.Random(cfg["seed"])
    header = system_header(mcfg, tok) if cfg["probe"].get("prefix", "none") == "mixed" else ""
    seqs = []
    for ti, text in enumerate(texts):
        prefix = header if header and rng.random() < 0.5 else ""
        for role in cfg["probe"]["roles"]:
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
    return s[: s.index(mcfg["probe_wrappers"]["user"][0] + marker)]


def extract_training_acts(cfg: dict, model, tok, seqs: list[dict], layers: list[int]):
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
    labels = [cfg["probe"]["roles"].index(s["role"]) for s in seqs]
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


def _eval(p, X, y):
    pr = predict_proba(p, X.float())
    nll = torch.nn.functional.nll_loss(torch.log(pr + 1e-12), y).item()
    pred = pr.argmax(-1)
    n = p["W"].shape[1]
    conf = torch.zeros(n, n, dtype=torch.long)
    for t, q in zip(y.tolist(), pred.tolist()):
        conf[t, q] += 1
    return {"acc": (pred == y).float().mean().item(), "nll": nll, "confusion": conf.tolist()}


def train_probes(cfg: dict, acts: dict, labels: list[int], text_ix: list[int]) -> dict:
    """Split base texts 60/20/20 (train/val/test); pick C on val; report test metrics."""
    rng = np.random.default_rng(cfg["seed"])
    uniq = rng.permutation(sorted(set(text_ix)))
    n = len(uniq)
    split = {t: ("train" if i < 0.6 * n else "val" if i < 0.8 * n else "test") for i, t in enumerate(uniq)}
    roles = cfg["probe"]["roles"]
    probes = {}
    for li, seq_acts in acts.items():
        parts = {k: ([], []) for k in ("train", "val", "test")}
        for a, lab, t in zip(seq_acts, labels, text_ix):
            parts[split[t]][0].append(a)
            parts[split[t]][1].append(torch.full((len(a),), lab))
        data = {k: (torch.cat(xs), torch.cat(ys)) for k, (xs, ys) in parts.items()}
        best = None
        for C in cfg["probe"]["C_grid"]:
            p = fit_lr(*data["train"], C=C, n_cls=len(roles))
            v = _eval(p, *data["val"])
            if best is None or v["nll"] < best[1]["nll"]:
                best = (p, v)
        p, v = best
        p.update(layer=li, roles=roles, val=v, test=_eval(p, *data["test"]),
                 n_train_tokens=len(data["train"][1]))
        probes[li] = p
        print(f"  layer {li:>2}: C={p['C']:.0e} val_acc={v['acc']:.3f} test_acc={p['test']['acc']:.3f}")
    return probes


def eval_layer(cfg: dict, layers: list[int]) -> int:
    el = cfg["probe"]["eval_layer"]
    return sorted(layers)[len(layers) // 2] if el == "mid" else int(el)


def run(cfg: dict, mcfg: dict, model, tok, out) -> dict:
    path = out / "probes.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    layers = M.probe_layers(model, cfg["probe"]["layer_stride"])
    texts = load_neutral_texts(cfg, tok)
    seqs = build_sequences(cfg, mcfg, tok, texts)
    acts, labels = extract_training_acts(cfg, model, tok, seqs, layers)
    probes = train_probes(cfg, acts, labels, [s["text_ix"] for s in seqs])
    bundle = {"probes": probes, "layers": layers, "eval_layer": eval_layer(cfg, layers), "n_texts": len(texts)}
    torch.save(bundle, path)
    summary = [{"layer": li, "C": p["C"], "val_acc": p["val"]["acc"], "test_acc": p["test"]["acc"],
                "test_confusion": p["test"]["confusion"], "roles": p["roles"]} for li, p in probes.items()]
    (out / "probe_metrics.json").write_text(json.dumps(summary, indent=1))
    return bundle
