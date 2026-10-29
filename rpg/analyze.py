"""Step 2 (benchmark score) tables and plots, per model and combined."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, MUTED = "#1f1f1e", "#6b6b66"
GROUPS = ["paper:user", "paper:assistant", "paper:tool", "paper:other", "attack", "native"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(s: pd.Series) -> dict:
    k, n = int(s.sum()), int(s.count())
    lo, hi = wilson(k, n)
    return {"n": n, "rate": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi}


def load(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-example table (Userness at the eval layer + labels + answers if generated), roleness, probe info."""
    meta = pd.DataFrame([{k: r[k] for k in ("ex_id", "doc_id", "payload", "placement", "style_id", "family", "label", "native")}
                         for r in map(json.loads, (out / "prompts.jsonl").read_text().splitlines())])
    rn = pd.read_parquet(out / "roleness.parquet")
    pm = json.loads((out / "probe_metrics.json").read_text())
    L = next(p["layer"] for p in pm if p.get("eval_layer"))
    core = rn[(rn.layer == L) & (rn.span == "injection_core")][["ex_id", "p_user", "p_tool"]]
    df = meta[meta.placement != "clean"].merge(core, on="ex_id").rename(columns={"p_user": "userness", "p_tool": "toolness"})
    df["group"] = np.where(df.native, "native", np.where(df.family == "attack", "attack", "paper:" + df.label.astype(str)))
    sc = out / "scores.jsonl"
    if sc.exists():
        s = pd.read_json(sc, lines=True)
        df = df.merge(s[s.placement != "clean"][["ex_id", "obeyed", "task_done", "finished"]], on="ex_id", how="left")
    return df, rn, {"eval_layer": L, "metrics": pm}


def benchmark_score(df: pd.DataFrame, seed: int = 0) -> dict:
    """Tag effect, style effect and score on non-native wrappers (native ones are real tags, not style)."""
    d = df[~df.native]
    wide = d.pivot_table(index=["doc_id", "payload", "style_id"], columns="placement", values="userness")
    diff = (wide["user"] - wide["tool"]).groupby(level="style_id").mean()
    per_style_tool = d[d.placement == "tool"].groupby("style_id").userness.mean()
    rng = np.random.default_rng(seed)
    boots = [diff.sample(len(diff), replace=True, random_state=rng.integers(1 << 31)).mean() for _ in range(1000)]
    tag = diff.mean()
    spread = per_style_tool.quantile(0.9) - per_style_tool.quantile(0.1)
    return {"tag_effect": tag, "tag_ci_lo": np.quantile(boots, 0.025), "tag_ci_hi": np.quantile(boots, 0.975),
            "style_effect_p90_p10": spread, "style_sd": per_style_tool.std(), "score": spread / tag if tag > 0 else np.inf,
            "userness_tool_mean": per_style_tool.mean(), "userness_user_mean": d[d.placement == "user"].userness.mean(),
            "n_styles": len(per_style_tool)}


def by_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pl, g), x in df.groupby(["placement", "group"]):
        r = {"placement": pl, "group": g, "n_styles": x.style_id.nunique(), "userness": x.userness.mean()}
        if "obeyed" in x and x.obeyed.notna().any():
            o = rate(x.obeyed.dropna().astype(bool))
            r.update(obey_rate=o["rate"], obey_lo=o["ci_lo"], obey_hi=o["ci_hi"], n_answers=o["n"],
                     task_done=x.task_done.dropna().astype(bool).mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    t["group"] = pd.Categorical(t.group, GROUPS, ordered=True)
    return t.sort_values(["placement", "group"])


def base_rates(out: Path) -> pd.DataFrame:
    s = pd.read_json(out / "scores.jsonl", lines=True)
    c = s[s.placement == "clean"]
    return pd.DataFrame([{"payload": p, **rate(x.obeyed.astype(bool))} for p, x in c.groupby("payload")])


def userness_vs_obey(df: pd.DataFrame, n_perm: int = 2000, seed: int = 0) -> dict:
    """Spearman across wrappers in the tool result (non-native): mean Userness vs obey rate."""
    d = df[(df.placement == "tool") & ~df.native & df.obeyed.notna()]
    per = d.assign(obeyed=d.obeyed.astype(float)).groupby("style_id").agg(userness=("userness", "mean"), obey=("obeyed", "mean"))
    if len(per) < 5 or per.obey.nunique() < 2:
        return {"rho": np.nan, "p_perm": np.nan, "n_styles": len(per)}
    rho = spearmanr(per.userness, per.obey).statistic
    rng = np.random.default_rng(seed)
    null = [spearmanr(per.userness, rng.permutation(per.obey.values)).statistic for _ in range(n_perm)]
    return {"rho": rho, "p_perm": (np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1), "n_styles": len(per)}


# ---------------------------------------------------------------- plots

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_groups(dfs: dict[str, pd.DataFrame], path: Path):
    fig, axes = plt.subplots(1, len(dfs), figsize=(4.2 * len(dfs), 3.6), squeeze=False, sharey=True)
    for ax, (m, df) in zip(axes[0], dfs.items()):
        per = df.groupby(["group", "placement", "style_id"]).userness.mean().reset_index()
        groups = [g for g in GROUPS if g in set(per.group)]
        for j, (pl, c, dx) in enumerate([("user", BLUE, -0.17), ("tool", ORANGE, 0.17)]):
            for i, g in enumerate(groups):
                y = per[(per.group == g) & (per.placement == pl)].userness
                x = i + dx + np.random.default_rng(i).uniform(-0.07, 0.07, len(y))
                ax.scatter(x, y, s=10, color=c, alpha=0.55, lw=0, label=pl if i == 0 else None)
                ax.plot([i + dx - 0.12, i + dx + 0.12], [y.mean()] * 2, color=INK, lw=1.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace("paper:", "") for g in groups], rotation=25, ha="right")
        ax.set_title(m, fontsize=10, color=INK)
        ax.set_ylim(-0.02, 1.02)
        _style(ax)
    axes[0][0].set_ylabel("Userness of the instruction")
    axes[0][0].legend(title="placement", frameon=False, fontsize=8, title_fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def run(cfg: dict, names: list[str]) -> None:
    root = Path(cfg["out_dir"])
    summ = root / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    dfs, headline, md = {}, [], ["# Step 2: results\n"]
    for m in names:
        out = root / m
        if not (out / "roleness.parquet").exists():
            continue
        df, _rn, info = load(out)
        dfs[m] = df
        df.to_csv(summ / f"{m}_examples.csv", index=False)
        L = info["eval_layer"]
        pinfo = next(p for p in info["metrics"] if p["layer"] == L)
        bs = benchmark_score(df)
        groups = by_group(df)
        h = {"model": m, "eval_layer": L, "probe_span_acc": pinfo["test_span_acc"], **bs}
        md += [f"## {m}\n", f"Eval layer {L}; probe test span accuracy {pinfo['test_span_acc']:.3f}.\n",
               "### Benchmark score (non-native wrappers)\n", _md(pd.DataFrame([bs])),
               "\n### By wrapper group\n", _md(groups)]
        if (out / "scores.jsonl").exists():
            corr = userness_vs_obey(df)
            br = base_rates(out)
            tool = df[(df.placement == "tool") & ~df.native].obeyed.dropna().astype(bool)
            user = df[(df.placement == "user") & ~df.native].obeyed.dropna().astype(bool)
            h.update(obey_tool=tool.mean(), obey_user=user.mean(), rho_userness_obey=corr["rho"], rho_p=corr["p_perm"])
            md += ["\n### Base rate (no instruction)\n", _md(br),
                   (f"\nSpearman (per wrapper, tool result, non-native) Userness vs obey rate: rho = {corr['rho']:.3f}, "
                   f"permutation p = {corr['p_perm']:.3g}, n = {corr['n_styles']} wrappers.\n")]
        headline.append(h)
        md.append("")
    if not dfs:
        return
    plot_groups(dfs, summ / "userness_by_group.png")
    hl = pd.DataFrame(headline)
    hl.to_csv(summ / "headline.csv", index=False)
    md[1:1] = ["## Headline\n", _md(hl), ""]
    (summ / "results.md").write_text("\n".join(md))
    print(f"wrote {summ / 'results.md'}")
