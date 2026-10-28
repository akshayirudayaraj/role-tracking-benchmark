"""Probe wrappers must match the model's real chat template, and spans must be locatable."""
import json

import pytest
from transformers import AutoTokenizer

from rpg.config import load_config, model_cfg
from rpg.dataset import build_examples, load_docs
from rpg.templating import (
    FETCH_TOOL,
    build_messages,
    char_spans_to_token_idx,
    locate_spans,
    render,
)

CFG = load_config("configs/base.yaml")
MARK = "Beginners BBQ Class Taking Place in Missoula!"


def _tok(name):
    try:
        return AutoTokenizer.from_pretrained(CFG["models"][name]["hf_id"])
    except OSError as e:  # gated / offline
        pytest.skip(str(e))


@pytest.mark.parametrize("name", list(CFG["models"]))
def test_wrappers_match_real_template(name):
    mcfg, tok = model_cfg(CFG, name), _tok(name)
    kw = mcfg.get("template_kwargs", {})
    convs = {
        "user": [{"role": "user", "content": MARK}],
        "system": [{"role": "system", "content": MARK}, {"role": "user", "content": "hi"}],
        "tool": [{"role": "user", "content": "go"},
                 {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "fetch_url", "arguments": {"url": "u"}}}]},
                 {"role": "tool", "name": "fetch_url", "content": MARK}],
        # Qwen3 reads `reasoning_content`, gpt-oss reads `thinking`.
        "thinking": [{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "ok", "reasoning_content": MARK, "thinking": MARK}],
    }
    for role, msgs in convs.items():
        if role not in mcfg["probe_wrappers"]:
            continue
        if role == "system" and "gpt-oss" in name:
            continue  # gpt-oss maps a system message to the developer role; probe uses the raw system role
        real = tok.apply_chat_template(msgs, tools=[FETCH_TOOL] if role == "tool" else None, tokenize=False, **kw)
        pre, suf = mcfg["probe_wrappers"][role]
        content = json.dumps(MARK)[1:-1] if role in mcfg.get("json_escape_roles", []) else MARK
        if role == "system":  # Llama prepends fixed date lines to every system message
            assert pre in real and (content + suf).strip() in real, f"{name}/system wrapper not found in:\n{real}"
            continue
        assert (pre + content + suf).strip() in real, f"{name}/{role} wrapper not found in:\n{real}"


@pytest.mark.parametrize("name", list(CFG["models"]))
def test_spans_locatable(name):
    """Every wrapper x payload x placement renders and its spans map to tokens (one document)."""
    mcfg, tok = model_cfg(CFG, name), _tok(name)
    cfg = {**CFG, "n_docs": 1, "gen_docs": 1}
    docs = {d["doc_id"]: d for d in load_docs(cfg)}
    for ex in build_examples(cfg):
        doc = docs[ex["doc_id"]]
        inj = ex["injection"]
        prompt = render(tok, mcfg, *build_messages(CFG, ex, doc, inj))
        spans = locate_spans(prompt, CFG, ex, doc, inj)
        _, toks = char_spans_to_token_idx(tok, prompt, spans)
        if inj is not None:
            assert spans["injection"][0] > spans["doc_body"][0]
            assert len(toks["injection_core"]) >= 3
