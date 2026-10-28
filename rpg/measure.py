"""Project eval prompts into role space: mean P(role) per span, at every probed layer."""
import pandas as pd
import torch
from tqdm import tqdm

from rpg import model as M
from rpg.probe import predict_proba


def run(cfg: dict, model, tok, rows: list[dict], bundle: dict, out) -> pd.DataFrame:
    path = out / "roleness.parquet"
    if path.exists():
        return pd.read_parquet(path)
    layers = bundle["layers"]
    roles = bundle["roles"]
    recs, cached_acts = [], {}
    bs = cfg["probe"]["batch_size"]
    for i in tqdm(range(0, len(rows), bs), desc="roleness"):
        chunk = rows[i:i + bs]
        # one position list per sequence = concatenation of all spans; remember slices
        positions, slices = [], []
        for r in chunk:
            pos, sl = [], {}
            for name, idx in r["span_tokens"].items():
                sl[name] = (len(pos), len(pos) + len(idx))
                pos += idx
            positions.append(pos)
            slices.append(sl)
        got = M.capture(model, tok, [r["input_ids"] for r in chunk], layers, positions)
        for b, r in enumerate(chunk):
            for li in layers:
                probs = predict_proba(bundle["probes"][li], got[li][b])
                for name, (a, z) in slices[b].items():
                    mean = probs[a:z].mean(0)
                    recs.append({"ex_id": r["ex_id"], "layer": li, "span": name, "n_tokens": z - a,
                                 **{f"p_{role}": mean[j].item() for j, role in enumerate(roles)}})
            if "injection" in slices[b]:
                a, z = slices[b]["injection"]
                cached_acts[r["ex_id"]] = got[bundle["eval_layer"]][b][a:z].half()
    df = pd.DataFrame(recs)
    df.to_parquet(path)
    torch.save(cached_acts, out / "injection_acts_eval_layer.pt")
    return df
