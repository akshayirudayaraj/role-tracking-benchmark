"""Probe wrappers must match the model's real chat template, and spans must be locatable."""
import json

import pytest
from transformers import AutoTokenizer

from rpg.config import load_config, model_cfg
from rpg.dataset import build_examples, load_docs
from rpg.probe import roles_for
from rpg.prompts import make_row
from rpg.templating import (
    FETCH_TOOL,
    char_spans_to_token_idx,
    locate_spans,
    render_prompt,
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
    """Every wrapper x payload x tag renders and its spans map to tokens (one document)."""
    mcfg, tok = model_cfg(CFG, name), _tok(name)
    cfg = {**CFG, "n_docs": 1, "gen_docs": 1}
    docs = {d["doc_id"]: d for d in load_docs(cfg)}
    tags = set(roles_for(CFG, mcfg)) | {"clean"}
    for ex in build_examples(cfg):
        if ex["placement"] not in tags:
            continue
        doc = docs[ex["doc_id"]]
        inj = ex["injection"]
        prompt = render_prompt(tok, CFG, mcfg, ex, doc, inj)
        spans = locate_spans(prompt, CFG, ex, doc, inj)
        _, toks = char_spans_to_token_idx(tok, prompt, spans)
        if inj is not None:
            assert spans["injection"][0] > spans["doc_body"][0]
            assert len(toks["injection_core"]) >= 3


@pytest.mark.parametrize("name", list(CFG["models"]))
def test_instruction_under_its_tag(name):
    """The plain instruction sits after the opening tag of its own role and before the next tag."""
    mcfg, tok = model_cfg(CFG, name), _tok(name)
    doc = load_docs(CFG, 1)[0]
    instr = CFG["payloads"]["end_word"]["instruction"]
    for tag in roles_for(CFG, mcfg):
        if tag == "system" and "gpt-oss" in name:
            continue  # gpt-oss renders a system message as a developer message
        ex = {"ex_id": tag, "placement": tag, "payload": "end_word"}
        prompt = render_prompt(tok, CFG, mcfg, ex, doc, instr)
        pre = mcfg["probe_wrappers"][tag][0].split("\n")[0].rstrip('"')
        i = prompt.rindex(instr)
        assert pre in prompt[:i], f"{name}/{tag}: opening tag missing before the instruction"
        last_pre = max(prompt.rfind(mcfg["probe_wrappers"][r][0].split("\n")[0].rstrip('"'), 0, i)
                       for r in roles_for(CFG, mcfg) if r != tag)
        assert prompt.rfind(pre, 0, i) >= last_pre, f"{name}/{tag}: another tag is closer to the instruction"


@pytest.mark.parametrize("name", list(CFG["models"]))
def test_single_layout(name):
    """`layout: single`: each tag's prompt is the probe format, and the instruction is locatable."""
    mcfg, tok = model_cfg(CFG, name), _tok(name)
    cfg = {**CFG, "layout": "single", "n_docs": 1}
    doc = load_docs(cfg, 1)[0]
    for ex in build_examples(cfg):
        if ex["placement"] not in roles_for(CFG, mcfg) or ex["style_id"] not in ("naive", "paper000"):
            continue
        row = make_row(cfg, mcfg, tok, ex, doc)
        pre, suf = mcfg["probe_wrappers"][ex["placement"]]
        assert row["prompt"].endswith(suf) and pre in row["prompt"]
        assert "request" not in row["span_tokens"] and len(row["span_tokens"]["injection_core"]) >= 3
