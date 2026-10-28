"""Aggregate scores + role projections into tables and plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PLACEMENT_COLORS = {"user": "#2a78d6", "user_doc": "#eb6834", "tool": "#1baf7a", "clean": "#8a8a85"}
INK, MUTED = "#1f1f1e", "#6b6b66"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_model(out: Path, roles: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scores = pd.read_json(out / "scores.jsonl", lines=True)
    rn = pd.read_parquet(out / "roleness.parquet")
    bundle = torch.load(out / "probes.pt", weights_only=False)
    return scores, rn, bundle


def per_example(scores: pd.DataFrame, rn: pd.DataFrame, layer: int) -> pd.DataFrame:
    inj = rn[(rn.layer == layer) & (rn.span == "injection")][["ex_id", "p_user", "p_tool", "p_assistant", "p_system"]]
    df = scores[scores.placement != "clean"].merge(inj, on="ex_id", how="left")
    df["userness"] = df["p_user"]
    df["outcome"] = np.select(
        [df.complied & df.task_done, df.complied & ~df.task_done, ~df.complied & df.task_done],
        ["both", "injection_only", "task_only"], "neither")
    return df


def cell_table(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pl, cue), g in df.groupby(["placement", "cue"]):
        k, n = int(g.complied.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"placement": pl, "cue": cue, "n": n, "compliance": k / n, "ci_lo": lo, "ci_hi": hi,
                     "task_done": g.task_done.mean(), "injection_only": (g.outcome == "injection_only").mean(),
                     "userness": g.userness.mean(), "userness_se": g.userness.std(ddof=1) / np.sqrt(n),
                     "toolness": g.p_tool.mean()})
    clean = scores[scores.placement == "clean"]
    if len(clean):
        k, n = int(clean.complied.sum()), len(clean)
        lo, hi = wilson(k, n)
        rows.append({"placement": "clean", "cue": "-", "n": n, "compliance": k / n, "ci_lo": lo, "ci_hi": hi,
                     "task_done": clean.drop_duplicates("ex_id").task_done.mean(), "injection_only": np.nan,
                     "userness": np.nan, "userness_se": np.nan, "toolness": np.nan})
    return pd.DataFrame(rows)


def gap_table(cells: pd.DataFrame) -> pd.DataFrame:
    c = cells[cells.placement != "clean"].pivot(index="cue", columns="placement", values=["userness", "compliance"])
    out = pd.DataFrame(index=c.index)
    for pl in ("tool", "user_doc"):
        if ("userness", pl) in c:
            out[f"userness_gap_{pl}"] = c[("userness", pl)] - c[("userness", "user")]
            out[f"compliance_gap_{pl}"] = c[("compliance", pl)] - c[("compliance", "user")]
    return out.reset_index()


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else np.nan


def correlation(df: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> dict:
    """Spearman(Userness, complied) over injected examples NOT in the genuine user turn.
    p-value from a permutation test that shuffles Userness *within payload x placement strata*, so a
    payload that is both easy to follow and user-sounding can't drive the result alone."""
    d = df[df.placement != "user"].dropna(subset=["userness"])
    x, y = d.userness.values, d.complied.astype(float).values
    rho = _rank_corr(x, y)
    rng = np.random.default_rng(seed)
    strata = [np.where((d.payload == p).values & (d.placement == q).values)[0]
              for p in d.payload.unique() for q in d.placement.unique()]
    null = []
    for _ in range(n_perm):
        xp = x.copy()
        for s in strata:
            xp[s] = x[rng.permutation(s)]
        null.append(_rank_corr(xp, y))
    null = np.array(null)
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1)) if not np.isnan(rho) else np.nan
    return {"spearman_rho": rho, "perm_p_within_payload_placement": p, "n": len(d)}


def validity(rn: pd.DataFrame, scores: pd.DataFrame, layer: int) -> pd.DataFrame:
    """Userness of non-injected spans: the user's own request and the document body, by placement.
    A valid probe should give high Userness to the request and, for tool placement, low Userness to the doc."""
    pl = scores.drop_duplicates("ex_id")[["ex_id", "placement"]]
    d = rn[(rn.layer == layer) & rn.span.isin(["request", "doc_body"])].merge(pl, on="ex_id")
    return d.groupby(["span", "placement"])[["p_user", "p_tool", "p_assistant", "p_system"]].mean().reset_index()


def layer_sensitivity(scores: pd.DataFrame, rn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for li in sorted(rn.layer.unique()):
        df = per_example(scores, rn, li)
        rows.append({"layer": li, **{k: v for k, v in correlation(df, n_perm=500).items()},
                     "userness_gap_tool_none": df[(df.placement == "tool")].userness.mean() - df[df.placement == "user"].userness.mean()})
    return pd.DataFrame(rows)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def plot_by_condition(all_cells: dict[str, pd.DataFrame], path: Path):
    models = list(all_cells)
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.2 * len(models)), squeeze=False)
    for r, m in enumerate(models):
        cells = all_cells[m]
        inj = cells[cells.placement != "clean"]
        cues = list(dict.fromkeys(inj.cue))
        pls = [p for p in ("user", "user_doc", "tool") if p in set(inj.placement)]
        w = 0.8 / len(pls)
        for c, (metric, lab) in enumerate([("compliance", "Compliance rate"), ("userness", "Mean Userness of injection")]):
            ax = axes[r, c]
            for i, pl in enumerate(pls):
                g = inj[inj.placement == pl].set_index("cue").reindex(cues)
                x = np.arange(len(cues)) + (i - (len(pls) - 1) / 2) * w
                ax.bar(x, g[metric], width=w * 0.92, color=PLACEMENT_COLORS[pl], label=pl)
                if metric == "compliance":
                    ax.errorbar(x, g[metric], yerr=[g[metric] - g.ci_lo, g.ci_hi - g[metric]], fmt="none", ecolor=INK, elinewidth=0.7, capsize=0)
                else:
                    ax.errorbar(x, g[metric], yerr=1.96 * g.userness_se, fmt="none", ecolor=INK, elinewidth=0.7, capsize=0)
            if metric == "compliance":
                base = cells[cells.placement == "clean"].compliance
                if len(base):
                    ax.axhline(base.iloc[0], color=MUTED, ls="--", lw=1)
                    ax.text(len(cues) - 0.5, base.iloc[0] + 0.02, "no-injection base rate", color=MUTED, fontsize=7, ha="right")
            ax.set_xticks(range(len(cues)))
            ax.set_xticklabels(cues, rotation=20, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f"{m} — {lab}", fontsize=10, color=INK, loc="left")
            _style(ax)
        axes[r, 0].legend(fontsize=8, frameon=False, title="placement", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scatter(all_cells: dict[str, pd.DataFrame], all_corr: dict[str, dict], path: Path):
    models = list(all_cells)
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[0, i]
        cells = all_cells[m][lambda d: d.placement != "clean"]
        for pl, g in cells.groupby("placement"):
            ax.scatter(g.userness, g.compliance, s=40, color=PLACEMENT_COLORS[pl], label=pl, edgecolor="white", lw=1.5)
            for _, row in g.iterrows():
                ax.annotate(row.cue, (row.userness, row.compliance), fontsize=6, color=MUTED, xytext=(3, 3), textcoords="offset points")
        c = all_corr[m]
        ax.set_title(f"{m}\nexample-level ρ={c['spearman_rho']:.2f} (perm p={c['perm_p_within_payload_placement']:.3f}, n={c['n']})",
                     fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Mean Userness of injected span (cell)", fontsize=8, color=MUTED)
        ax.set_ylabel("Compliance rate (cell)", fontsize=8, color=MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.03)
        _style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False, title="placement", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def run(cfg: dict, model_names: list[str]):
    root = Path(cfg["out_dir"])
    summ = root / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    all_cells, all_corr, md = {}, {}, []
    for m in model_names:
        out = root / m
        if not (out / "scores.jsonl").exists():
            continue
        scores, rn, bundle = load_model(out, cfg["probe"]["roles"])
        L = bundle["eval_layer"]
        df = per_example(scores, rn, L)
        df.to_csv(summ / f"{m}_examples.csv", index=False)
        cells = cell_table(df, scores)
        gaps = gap_table(cells)
        corr = correlation(df)
        val = validity(rn, scores, L)
        lay = layer_sensitivity(scores, rn)
        pm = json.loads((out / "probe_metrics.json").read_text())
        probe_df = pd.DataFrame([{"layer": p["layer"], "C": p["C"], "val_acc": p["val_acc"], "test_acc": p["test_acc"]} for p in pm])
        all_cells[m], all_corr[m] = cells, corr
        for name, t in [("cells", cells), ("gaps", gaps), ("validity", val), ("layers", lay), ("probe", probe_df)]:
            t.to_csv(summ / f"{m}_{name}.csv", index=False)
        md += [f"## {m} (eval layer {L}, {bundle['n_texts']} probe texts)",
               "### Probe held-out accuracy (split by base text; chance = 0.25)", _md(probe_df),
               "### Cells", _md(cells), "### Role-perception gap (placement − genuine user turn, same cue)", _md(gaps),
               "### Userness ↔ compliance", "```\n" + json.dumps(corr, indent=1) + "\n```",
               "### Probe validity: non-injected spans", _md(val), "### Layer sensitivity", _md(lay), ""]
    (summ / "tables.md").write_text("\n\n".join(md))
    if all_cells:
        plot_by_condition(all_cells, summ / "by_condition.png")
        plot_scatter(all_cells, all_corr, summ / "userness_vs_compliance.png")
    print(f"wrote {summ}")
