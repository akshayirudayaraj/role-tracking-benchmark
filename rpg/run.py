"""End-to-end pipeline. Every stage caches to <out_dir>/<model>/ and is skipped if its output exists.

    python -m rpg.run configs/smoke.yaml            # all stages, all run_models
    python -m rpg.run configs/pilot.yaml --models qwen3-8b --stages probe,measure
    python -m rpg.run configs/pilot.yaml --stages analyze
"""
import argparse
import gc
import json

import torch

from rpg import analyze, generate, measure, probe, prompts, report, score
from rpg import model as M
from rpg.config import load_config, model_cfg, out_dir
from rpg.dataset import load_docs

STAGES = ["probe", "measure", "generate", "score", "analyze"]
MODEL_STAGES = {"probe", "measure", "generate", "score"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--models", default=None, help="comma list; default = run_models from config")
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    cfg = load_config(args.config)
    stages = args.stages.split(",")
    names = args.models.split(",") if args.models else cfg["run_models"]
    torch.manual_seed(cfg["seed"])

    for name in names:
        if not set(stages) & MODEL_STAGES:
            break
        mcfg, out = model_cfg(cfg, name), out_dir(cfg, name)
        print(f"=== {name} -> {out}")
        model, tok = M.load(mcfg)
        if set(stages) & {"measure", "generate", "score"}:
            rows = prompts.build(cfg, mcfg, tok, out)
            print(f"{len(rows)} prompts; median length {sorted(len(r['input_ids']) for r in rows)[len(rows) // 2]} tokens")
        if set(stages) & {"probe", "measure"}:
            bundle = probe.run(cfg, mcfg, model, tok, out)
        if "measure" in stages:
            measure.run(cfg, model, tok, rows, bundle, out)
        if "generate" in stages:
            gens = generate.run(cfg, mcfg, model, tok, rows, out)
        if "score" in stages:
            gens = generate.run(cfg, mcfg, model, tok, rows, out)  # cached: just loads
            docs = {d["doc_id"]: d for d in load_docs(cfg)}
            recs = score.score(cfg, rows, gens, docs)
            (out / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "probe" in stages:
        report.probes(cfg, names)
    if "analyze" in stages:
        analyze.run(cfg, names)


if __name__ == "__main__":
    main()
