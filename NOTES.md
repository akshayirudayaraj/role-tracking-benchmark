# Source notes (what matters for the benchmark)

## 1. Blogpost — "A mechanistic explanation of prompt injection" (LessWrong)
- Claim: models infer *who is speaking* from **style**, not from the role tag. Injected text that
  "sounds like" the user (or like the model's own CoT) inherits that role's authority.
- **Probe method (content-controlled):** take neutral web text (C4 / Dolma) with no inherent role,
  wrap the *identical* text in each role's tags, train a linear probe on mid-layer activations to
  predict the role tag per token, **excluding tag tokens**. Training on real conversations would let
  the probe learn "question-like phrasing" instead of role. Output = per-token P(role);
  *Userness* = P(user), *CoTness* = P(cot).
- **Evidence:** (a) destyling forged CoT (same meaning, no CoT-typical wording) cut CoT-Forgery
  ASR ~61% → ~10%; (b) 212 prefix variants ("Tool output:", "The below is from a user:", fake
  templates…) before an injected command in a coding agent: higher Userness of the injected span ↔
  higher ASR (plotted, no coefficient given).
- gpt-oss-20b has the cleanest linear role separability; user vs assistant separable in every model
  tested; `<think>` less consistent.
- Post argues static benchmarks only catch known attacks → the useful thing to benchmark is the
  *mechanism* (perceived role), not a fixed attack list. This is the gap our benchmark targets.

## 2. Paper (arXiv 2603.12277) + code (`external/prompt-injection-as-role-confusion`)
Reused / mirrored:
- `demo/role-probe-demo.ipynb`, `utils/probes.py::run_and_export_states`: activations are the
  **output of `post_attention_layernorm`** ("pre-MLP" hidden state) per layer — we hook the same
  module (`model.model.layers[i].post_attention_layernorm`), which exists for gpt-oss, Qwen3, Llama.
- Probe = L2 multinomial logistic regression, `C = 5e-3`, 90/10 split **by prompt**, every 4th layer,
  test at the middle probed layer (`TEST_LAYER_IX = sorted(layers)[len//2]`). They use cuML; we use
  an equivalent full-batch torch L-BFGS LR (same objective) to avoid a RAPIDS dependency.
- `utils/role_templates.py::render_single_*`: per-model single-role wrappers. Notably **Qwen3's tool
  role is a `user` turn containing `<tool_response>` tags** — tool output is architecturally
  user-role in Qwen. We keep this (it's part of what the benchmark should expose).
- Role space at eval: `['user','assistant','tool']` (+system/cot for gpt-oss). We use
  `system,user,assistant,tool` for every model so numbers are comparable.
- `experiments/agent-injections/config/injection-templates.yaml`: ~120 style wrappers
  (`"User message: \"[CONTENT]\""`, `<|im_start|>user…`, `[TOOL]…[/TOOL]`, `**Data below**:` …) each
  labelled user/assistant/tool. Our style-cue axis is a small, principled subset of this list.
- Userness of an injection = mean P(user) over the injected-span tokens (`flag_injection` +
  `run_projections` in `agent-injections/01-…ipynb`); they also report user−tool difference.
- Their training sequences have no system prompt; at test time they prepend the model's default
  system prompt (`config/probe.yaml: test_prefix`). We do the same (position/prefix shift caveat).

## 3. V-Steer (arXiv 2607.26228, `external/v-steer`)
- Not a residual direction: multiplies cached **V vectors** on token spans after prefill —
  privileged span ×(1+γA), conflicting span ×(1−γB), γA=2.5, γB=0.75 — only at "bad" heads chosen by
  direct logit attribution on the first predicted token (`dla.py`, `steering.py::steer_past_kv_V`,
  `generate_hf_with_v_steering`). Llama-3.1 / Qwen2.5 only; span locators don't handle Harmony/Qwen3.
- Big gains on Control Illusion (≈10% → ≈80% primary-constraint) but **injected tool-use stays ≈0%**
  (Tab 11), attributed to tool formatting — directly relevant: tool-channel injection is where IH
  steering fails, which our placement axis isolates.
- Side effects measured (MMLU −8.5 at defaults, IFEval, BBH) → any intervention arm must also
  report task completion, not just compliance.
- Reuse for stretch arm: `steer_past_kv_V(γA=0, γB>0)` on the injected span is a ready
  "suppress injection" baseline; a Userness-direction ablation needs our own prefill hook.

## 4. Control Illusion (arXiv 2502.15851)
- 6 **mutually exclusive, programmatically checkable** constraint pairs (English/French, all-caps/
  lowercase, word count, sentence count, keyword include/exclude, keyword frequency) × 100 IFEval
  tasks. Scoring is fully programmatic (after LLM-stripping of conflict acknowledgements).
- Metrics worth borrowing: R1/R2/R3 (primary-only / secondary-only / neither), PAR = R1/(R1+R2),
  a **same-channel / no-conflict baseline** for an instruction's base rate, order swaps.
- Finding: system/user separation barely works (PAR often <50%), but **social framing inside the
  user turn** (CEO vs intern, consensus) moves obedience a lot → cues beat channels. Supports our
  style-cue axis.

## 5. IHEval (NAACL 2025, arXiv 2502.08745)
- 9 tasks / 4 categories incl. **tool use**: instruction injected into a tool return
  (`get_webpage_text_content` → "translate to Spanish" while user asked for verb extraction).
- Settings: aligned / conflict / **reference** (everything merged in one user message) — the
  reference condition is exactly our "genuine user turn" control. Metric Δ vs reference.
- Findings: every model degrades under conflict; injected-tool conflict ≈0 for Llama-3.1-8B;
  **stronger wording in the low-priority input lowers scores** (style > priority).
- Data with `constraint_substring` spans bundled in `external/v-steer/ih_eval/benchmark/tool-use/`.

## Design consequences (what changed vs. the starting plan)
1. **Add a "no injection" baseline** per payload (Control Illusion base rate): e.g. all-caps or
   French could occur spontaneously; compliance is reported against this floor.
2. **Joint outcome, not just compliance** (Control Illusion R1/R2/R3, V-Steer side-effects): each
   example is classed `injection-only / both / task-only / neither`.
3. **Payloads are additive, not conflicting** (end-word, code, all-caps, French, start-phrase), so
   "both" is possible; that separates *role confusion* (treating the doc as an instruction source)
   from *hijacking* (abandoning the task). IHEval/Control-Illusion use exclusive pairs; ours trades
   that for a cleaner probe target. Noted as a limitation.
4. **Genuine-user-turn placement = IHEval "reference"**: same injected string, appended after the
   document inside the user's own message. Gap = Userness(tool/doc) − Userness(user) per cue.
5. **Fake chat-template cue is split** into `fake_template_native` (the model's own role tokens —
   which HF tokenizers turn into *real* special tokens, i.e. a structural escape, not a style cue)
   vs. `fake_template_text` (generic `<user>…</user>`). Conflating them would overstate "style".
6. **Built-in probe validity checks** beyond held-out accuracy: Userness of the *non-injected*
   document body by placement (should be low in tool, high in user), and of the user's real request.
7. **Model-general templating**: eval prompts use each model's `apply_chat_template`; spans are found
   by character offsets → token offsets, so no per-model regex labellers are needed. Probe wrappers
   are checked against the real template by a unit test.
