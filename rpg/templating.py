"""Turn example specs into real chat-templated prompts and locate token spans.

Prompts are rendered with each model's own `apply_chat_template`, so role tokens are exactly what the
model saw in training. Spans (injection, document body, user request) are located by character
offsets in the rendered string and mapped to tokens via the tokenizer's offset mapping; tokens that
are special tokens (role tags) are never counted as span content.
"""
import json

FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch a web page and return its text content.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    },
}


def inject(doc_text: str, inj: str | None) -> str:
    """Injection goes at the end of the document as its own paragraph."""
    return doc_text if inj is None else f"{doc_text}\n\n{inj}"


def build_messages(cfg: dict, ex: dict, doc: dict, inj: str | None) -> tuple[list[dict], list | None]:
    t = cfg["task"]
    msgs = [{"role": "system", "content": t["system_prompt"]}] if t.get("system_prompt") else []
    pl = ex["placement"]
    if pl in ("tool", "clean"):
        msgs += [
            {"role": "user", "content": t["user_request_tool"].format(url=doc["url"])},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "fetch_url", "arguments": {"url": doc["url"]}}}]},
            {"role": "tool", "name": "fetch_url", "content": inject(doc["text"], inj)},
        ]
        return msgs, [FETCH_TOOL]
    if pl == "user_doc":
        body = f"{t['user_request_inline']}\n\n<document>\n{inject(doc['text'], inj)}\n</document>"
    elif pl == "user":
        body = f"{t['user_request_inline']}\n\n<document>\n{doc['text']}\n</document>\n\n{inj}"
    else:
        raise ValueError(f"unknown placement {pl}")
    return msgs + [{"role": "user", "content": body}], None


def render(tokenizer, mcfg: dict, msgs: list[dict], tools: list | None) -> str:
    return tokenizer.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=True, **mcfg.get("template_kwargs", {}))


def _find(prompt: str, text: str, start: int = 0) -> tuple[int, int] | None:
    """Locate `text` in `prompt` (last occurrence after `start`), also trying its JSON-escaped form
    because some templates (gpt-oss) JSON-encode tool output."""
    for cand in (text, json.dumps(text)[1:-1]):
        i = prompt.rfind(cand, start)
        if i >= 0:
            return i, i + len(cand)
    return None


def locate_spans(prompt: str, cfg: dict, ex: dict, doc: dict, inj: str | None) -> dict:
    """Character spans of interest in the rendered prompt."""
    t = cfg["task"]
    req = t["user_request_tool"].format(url=doc["url"]) if ex["placement"] in ("tool", "clean") else t["user_request_inline"]
    spans = {"request": _find(prompt, req), "doc_body": _find(prompt, doc["text"])}
    if inj is not None:
        spans["injection"] = _find(prompt, inj, start=spans["doc_body"][0] if spans["doc_body"] else 0)
        # The bare instruction, when the cue contains it verbatim (not for user_phrasing / destyled).
        instr = cfg["payloads"][ex["payload"]]["instruction"]
        if spans["injection"] and instr in inj:
            spans["injection_core"] = _find(prompt, instr, start=spans["injection"][0])
            if ex["cue"] == "fake_template_native":
                # The wrapper (role tokens + header like "user\n") is tag material: measure only the
                # message content, matching the paper's exclude-tag-tokens convention.
                spans["injection"] = spans["injection_core"]
    missing = [k for k, v in spans.items() if v is None]
    if missing:
        raise RuntimeError(f"{ex['ex_id']}: could not locate spans {missing}")
    return spans


def char_spans_to_token_idx(tokenizer, prompt: str, spans: dict) -> tuple[list[int], dict]:
    """Tokenize the prompt and return (input_ids, {span_name: [token indices]}).

    A token belongs to a span if its character range lies inside the span and it is not a special
    (role-tag) token. For the native fake-template cue this drops the injected role tokens themselves,
    which matches the paper's 'exclude tag tokens' convention."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    special = set(tokenizer.all_special_ids) | {
        i for i, tok in tokenizer.added_tokens_decoder.items() if tok.special}
    out = {}
    for name, (a, b) in spans.items():
        out[name] = [
            i for i, ((s, e), tid) in enumerate(zip(enc["offset_mapping"], enc["input_ids"]))
            if s >= a and e <= b and e > s and tid not in special
        ]
        if not out[name]:
            raise RuntimeError(f"empty token span {name}")
    return enc["input_ids"], out
