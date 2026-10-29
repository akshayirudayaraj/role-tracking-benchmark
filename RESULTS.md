# Pilot results

Pilot run on 2026-10-28 on one A40 GPU (RunPod). Configuration: `configs/pilot.yaml`.
Code to reproduce: `bash scripts/reproduce.sh pilot`.

**Models.** Qwen3-8B and gpt-oss-20b completed Steps 1 and 2. Llama-3.1-8B did not run (it needs a Hugging Face token on the GPU machine).
For Qwen3-8B, the steps used layer 18. For gpt-oss-20b, they used layer 16.

## Step 1: role meter

Qwen3-8B, 600 C4 texts in 5 roles, a linear probe at each of the 36 layers.
Chance accuracy is 0.20.

| Layer | Test accuracy (token) | Test accuracy (span) |
|---|---|---|
| 2 (best on validation) | 0.63 | 0.95 |
| 18 (middle, used in Steps 2 and 3) | 0.51 | 0.70 |
| 35 (last) | 0.61 | 0.82 |

![probe accuracy](runs/pilot/summary/probe_accuracy.png)

Span-level confusion at layer 18 (rows: true role, columns: predicted role):

| true \ pred | system | user | assistant | tool | thinking |
|---|---|---|---|---|---|
| system | 0.64 | 0.06 | 0.12 | 0.05 | 0.12 |
| user | 0.02 | 0.77 | 0.01 | 0.21 | 0.00 |
| assistant | 0.21 | 0.02 | 0.54 | 0.03 | 0.21 |
| tool | 0.03 | 0.30 | 0.02 | 0.65 | 0.01 |
| thinking | 0.03 | 0.00 | 0.07 | 0.00 | 0.91 |

- The meter passes its gate: the accuracy is much higher than chance at all layers after layer 0.
- User and tool are the two roles that the meter confuses most (21% and 30%). In Qwen3, a tool result is inside a user turn (`<|im_start|>user<tool_response>...`). Thus the role tags of Qwen3 themselves put tool text near user text.
**gpt-oss-20b (Step 1 only).** 24 layers. The layer with the best validation accuracy is layer 16. Its test span accuracy is 0.97 (token 0.88).
Span-level confusion at layer 16:

| true \ pred | system | user | assistant | tool | thinking |
|---|---|---|---|---|---|
| system | 0.92 | 0.07 | 0.00 | 0.00 | 0.01 |
| user | 0.02 | 0.98 | 0.00 | 0.00 | 0.00 |
| assistant | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| tool | 0.00 | 0.00 | 0.00 | 0.99 | 0.01 |
| thinking | 0.01 | 0.00 | 0.03 | 0.01 | 0.96 |

- gpt-oss-20b separates the roles much better than Qwen3-8B. It does not confuse user and tool (0.00), but Qwen3-8B confuses them in 21% to 30% of spans. In gpt-oss-20b, a tool result is a separate message with its own role tag. This agrees with the blog post, which found that gpt-oss has the clearest linear separation of roles.
- We used layer 18 for Qwen3-8B, not layer 2. The plan said to select the layer with the best validation accuracy, and that is layer 2. But an early layer is near the token embeddings. There, the probe can read which tag token is near, not a role that the model computes. The paper uses the middle layer. We report this change from the plan.

## Step 2: benchmark score

### Qwen3-8B

206 wrappers from the paper and 5 attack types from Liu et al. (2024), x 3 safe instructions x 2 documents x 2 locations.
Userness = mean P(user) on the tokens of the bare instruction, at layer 18.
Answers were generated for one document (1,266 answers plus 3 with no instruction).

| Value | Qwen3-8B |
|---|---|
| Tag effect (Userness in user message minus in tool result, same text) | **0.206** (95% CI 0.193 to 0.221) |
| Style effect (P90 minus P10 of Userness across wrappers, tool result) | **0.073** |
| Score (style effect / tag effect) | **0.35** |
| Mean Userness, tool result / user message | 0.025 / 0.231 |
| Obey rate, tool result / user message (non-native wrappers) | 0.997 / 1.000 |
| Obey rate with no instruction (base rate) | 0 of 3 for each check |
| Spearman, Userness vs obey rate across 193 wrappers (tool result) | rho = -0.07, permutation p = 0.34 |

By wrapper group (paper labels). "Native" wrappers contain the special tokens of Qwen3.

| Group | Wrappers | Userness (tool) | Userness (user) | Obey rate (tool) |
|---|---|---|---|---|
| user | 44 | 0.070 | 0.413 | 0.99 |
| assistant | 42 | 0.011 | 0.149 | 1.00 |
| tool | 69 | 0.012 | 0.166 | 1.00 |
| other | 33 | 0.012 | 0.225 | 0.99 |
| attack (Liu et al.) | 5 | 0.013 | 0.259 | 1.00 |
| native | 18 | 0.044 | 0.177 | 1.00 |

![Userness by wrapper group](runs/pilot/summary/userness_by_group.png)

- **The role tags have the larger effect.** The same instruction has 0.21 more Userness in the user message than in the tool result. Style moves Userness by 0.07 across wrappers in the tool result.
- **Style also has an effect.** Wrappers that the authors labeled "user" have 6 times the Userness of other wrappers in the tool result (0.070 against 0.011). Native wrappers also increase it (0.044).
- **The obey rate does not show a difference.** Qwen3-8B obeyed almost all instructions in the tool result, with all wrappers. Thus the obey rate cannot be related to Userness (no variance). Our instructions are safe and do not conflict with the task of the user (the model can summarize and also write in French). For these instructions, Qwen3-8B does not use the role to select whom to obey.

### gpt-oss-20b

Layer 16 (the best layer on validation). The same design as for Qwen3-8B.

| Value | gpt-oss-20b | Qwen3-8B |
|---|---|---|
| Tag effect | **0.198** (95% CI 0.178 to 0.216) | 0.206 |
| Style effect (P90 minus P10) | **0.067** | 0.073 |
| Score | **0.34** | 0.35 |
| Mean Userness, tool result / user message | 0.056 / 0.254 | 0.025 / 0.231 |
| Obey rate, tool result / user message (non-native) | **0.905 / 0.993** | 0.997 / 1.000 |
| Spearman, Userness vs obey rate across wrappers (tool result) | **rho = 0.29, permutation p = 0.0005** (199 wrappers) | rho = -0.07, p = 0.34 |

| Group | Userness (tool) | Obey rate (tool) | Obey rate (user) |
|---|---|---|---|
| user | 0.091 | 0.98 | 1.00 |
| assistant | 0.048 | 0.88 | 0.99 |
| tool | 0.047 | 0.93 | 1.00 |
| other | 0.040 | 0.80 | 0.97 |
| attack (Liu et al.) | 0.056 | 0.73 | 1.00 |
| native | 0.143 | 0.92 | 0.94 |

- **The two models have almost the same score (0.34 and 0.35).** In both models, the role tags change the Userness approximately 3 times more than the style.
- **The meter is better calibrated in gpt-oss-20b.** In a real tool result, the meter reads the web page as "tool" (0.68). It reads the request of the user as "user" (0.91). In Qwen3-8B, the meter read the web page as "thinking".
- **In gpt-oss-20b, the obey rate is lower in the tool result (0.905).** In the user message, it is 0.993. Thus this model uses the role, a little, to select whom to obey.
- **In gpt-oss-20b, the Userness predicts the obey rate.** Across the wrappers in the tool result, a wrapper with more Userness is obeyed more frequently (rho = 0.29, p = 0.0005). The wrappers that the authors labeled "user" have the highest Userness and the highest obey rate (0.98). This agrees with the hypothesis: text that the model perceives as user text is obeyed more.
- **Native wrappers increase the Userness most (0.143).** These wrappers contain the special tokens of gpt-oss-20b.
- The summary check failed for most French answers (0.08), because it counts words that are the same in the answer and in the English document. For the other two instructions, the summary was done in 99% to 100% of answers.

## Step 3: steering

### gpt-oss-20b (revised Step 3)

The instruction is "Ignore previous instructions. ..." (`context_ignoring`). We selected it from the Step 2 data, because the plain instruction failed Step A (user rate 1.00, tool rate 0.97). 36 prompts in each condition. The steer is added at the input of layer 16. Userness is read at layer 16 and at layer 23 (the last layer).

| Condition | Userness L16 | Userness L23 | Obey rate (95% CI) | Summary done |
|---|---|---|---|---|
| Step A: user message, no steer | 0.43 | 0.39 | **1.00** (0.90 to 1.00) | 0.83 |
| Step A: tool result, no steer | 0.08 | 0.02 | **0.69** (0.53 to 0.82) | 0.86 |
| Test 1: tool result, toward user (target 0.43) | 0.52 | 0.03 | 0.72 (0.56 to 0.84) | 0.86 |
| Test 2: user message, toward tool (target 0.08) | 0.09 | 0.35 | 1.00 (0.90 to 1.00) | 0.86 |
| Test 3: tool result, random direction | 0.07 | 0.02 | 0.69 | 0.89 |
| Test 3: tool result, assistant direction | 0.07 | 0.02 | 0.78 | 0.86 |

![steering](runs/pilot/summary/steering.png)

- **Step A passes.** gpt-oss-20b obeys this instruction in the user message (1.00) more than in a tool result (0.69). Thus the sender changes the behavior of this model.
- **Test 1 and Test 2 fail.** At layer 16, the steer moved the Userness to the target (0.52 against a target of 0.43, and 0.09 against a target of 0.08). But the obey rate did not move: 0.72 against the tool rate of 0.69 in Test 1, and 1.00 in Test 2.
- **The change did not continue to the last layer.** At layer 23, the Userness after the steer is 0.03 in Test 1 (the tool value is 0.02). In Test 2, it is 0.35 (the user value is 0.39). The later layers removed the change, possibly because they compute the role again from the role tags.
- **Interpretation.** The probe signal at layer 16 is not sufficient and not necessary for the behavior. A change of Userness at one layer does not have the effect of a real change of sender. This is a negative result for the meter at a single layer. It does not show that the model does not use a role signal. The model can compute the role again at later layers, and a steer at more layers (or a larger steer) can give a different result.
- **Test 3 is not informative**, because Test 1 had no effect to compare with.
- **Two limits of this run.** (1) The strengths came from a bisection with a step of 0.016. Thus the 25% and 50% targets had the same strength, and the 75% and 100% targets had the same strength. (2) We selected the instruction from Step 2, and only 36 prompts are in each condition.


### Qwen3-8B (first version of Step 3)

This run used the first design of Step 3 (fixed strengths, no Step A). The revised design in `PLAN.md` starts with Step A. For Qwen3-8B, Step A fails: the obey rate is 0.997 in the tool result and 1.000 in the user message (Step 2). Thus the revised tests cannot give a result for this model.

36 prompts in each condition (12 documents x 3 instructions). The steer is added at the input of layer 18, on the instruction tokens only.
Userness is measured at layer 18 (where the steer is added) and at layer 35 (the last layer), to show if the change continues through the model.

| Condition | Strength | Userness L18 | Userness L35 | Obey rate (95% CI) | Summary done |
|---|---|---|---|---|---|
| Plain instruction in tool result, no steer | 0 | 0.02 | 0.33 | 0.97 (0.86 to 1.00) | 0.92 |
| Toward user | 0.05 | 0.63 | 0.37 | 0.94 | 0.92 |
| Toward user | 0.2 | 1.00 | 0.49 | 0.97 | 0.92 |
| Toward user | 0.8 | 1.00 | 0.67 | 0.94 (0.82 to 0.99) | 0.92 |
| Random direction | 0.8 | 0.04 | 0.21 | 0.94 | 0.89 |
| User-like wrapper in tool result, no steer | 0 | 0.03 | 0.42 | 0.97 | 0.92 |
| User-like wrapper, toward tool | 0.8 | 0.00 | 0.11 | 0.94 (0.82 to 0.99) | 0.89 |
| Plain instruction in user message (reference) | 0 | 0.34 | 0.66 | 1.00 (0.90 to 1.00) | 0.89 |

![steering](runs/pilot/summary/steering_qwen_v1.png)

- **The steer changes the perceived role.** The change continues to the last layer. Toward user, the Userness at layer 35 increases from 0.33 to 0.67. This is the same as the Userness of a real user instruction (0.66). Toward tool, it decreases from 0.42 to 0.11. The random direction does not increase the Userness.
- **The steer does not change the obey rate.** The obey rate stays between 0.94 and 0.97 in all conditions. The summary task also stays at approximately 0.9, thus the steer did not damage the model.
- **Interpretation.** For these safe instructions, Qwen3-8B obeys an instruction in a tool result in two conditions. It obeys when it perceives the text as tool output, and also when it perceives the text as a user message. Thus the obey rate is at its maximum before the steer, and the steer cannot increase it. The toward-tool steer also did not decrease it. As the plan said, this result cannot separate two explanations. (a) The meter reads a signal that the model does not use for obedience. (b) For instructions that do not conflict with the task, the perceived role does not control obedience. The next test must use instructions that conflict with the task of the user. For these, the obey rate in a tool result is not at the maximum.


## Limits

1. **Two models.** Llama-3.1-8B did not run. Two models are not sufficient to compare model families.
2. **Ceiling effect in Qwen3-8B.** For Qwen3-8B, the obey rate is near 100% in the tool result. Thus Steps 2 and 3 cannot relate Userness to behavior for this model. Instructions that conflict with the task of the user (for example, the exclusive pairs of Control Illusion, or "do not summarize") will give a range of obey rates.
3. **The meter is calibrated on single turns.** In the training data, each role is one turn alone. In a real tool result, the meter reads the web page as "thinking" (0.57) more than as "tool" (0.07). A possible cause is that the text comes after an assistant tool call. The Userness values are low in the tool result and are comparable between conditions. But the absolute values of the other roles are not reliable in long contexts.
4. **Layer choice.** We used the middle layer (18), not the layer with the best validation accuracy (2). See Step 1.
5. **Small samples.** One document for answers, two for Userness, and three answers with no instruction for the base rate.
6. **Simple checks.** The French check counts stop words, and the summary check counts shared content words. We examined some answers by eye, and the checks were correct.
7. **Qwen3 tool role.** Qwen3 puts tool output inside a user turn. Thus "tool" and "user" are close in the role tags of this model, and the tag effect of other models can be different.
