"""Step 1 report: probe accuracy by layer and the confusion matrix at the eval layer, for each model."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]


def _load(root: Path, names: list[str]) -> dict[str, list[dict]]:
    res = {}
    for m in names:
        f = root / m / "probe_metrics.json"
        if f.exists():
            res[m] = json.loads(f.read_text())
    return res


def probes(cfg: dict, names: list[str]) -> None:
    root = Path(cfg["out_dir"])
    metrics = _load(root, names)
    if not metrics:
        return
    summ = root / "summary"
    summ.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (m, pm) in enumerate(metrics.items()):
        n_layers_frac = [p["layer"] for p in pm]
        c = PALETTE[i % 3]
        ax.plot(n_layers_frac, [p["test_span_acc"] for p in pm], marker="o", lw=2, ms=5, color=c, label=f"{m} (span)")
        ax.plot(n_layers_frac, [p["test_acc"] for p in pm], lw=1.5, ls="--", color=c, label=f"{m} (token)")
        ax.axhline(1 / len(pm[0]["roles"]), color=PALETTE[i % 3], lw=1, ls=":", alpha=0.6)
    ax.set(xlabel="layer", ylabel="held-out accuracy", ylim=(0, 1.02), title="Role probe accuracy (dotted = chance)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(summ / "probe_accuracy.png", dpi=150)
    plt.close(fig)

    md = ["# Step 1: role probes\n"]
    for m, pm in metrics.items():
        roles = pm[0]["roles"]
        best = max(pm, key=lambda p: p["test_span_acc"])
        mid = next((p for p in pm if p.get("eval_layer")), pm[len(pm) // 2])
        md.append(f"## {m}\n")
        md.append(f"Chance = {1 / len(roles):.2f}. Best layer {best['layer']} (span acc {best['test_span_acc']:.3f}); "
                  f"eval layer {mid['layer']} (span {mid['test_span_acc']:.3f}, token {mid['test_acc']:.3f}).\n")
        md.append("| layer | C | val acc (token) | test acc (token) | test acc (span) |\n|---|---|---|---|---|")
        md += [f"| {p['layer']} | {p['C']:.0e} | {p['val_acc']:.3f} | {p['test_acc']:.3f} | {p['test_span_acc']:.3f} |"
               for p in pm]
        for key, label in [("test_confusion", "token"), ("test_span_confusion", "span")]:
            md += _confusion_md(np.array(mid[key], dtype=float), roles, mid["layer"], label)
        md.append("")
    (summ / "probes.md").write_text("\n".join(md))
    print(f"wrote {summ / 'probes.md'} and probe_accuracy.png")


def _confusion_md(conf: np.ndarray, roles: list[str], layer: int, label: str) -> list[str]:
    conf = conf / conf.sum(1, keepdims=True)
    head = f"\n{label.capitalize()}-level confusion at eval layer {layer} (rows = true role, cols = predicted):\n"
    return [head, "| true \\ pred | " + " | ".join(roles) + " |\n|---|" + "---|" * len(roles),
            *[f"| {r} | " + " | ".join(f"{v:.2f}" for v in row) + " |" for r, row in zip(roles, conf)]]

