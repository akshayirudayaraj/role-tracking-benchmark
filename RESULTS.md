# Pilot results

Pilot runs on 2026-10-28 and 2026-10-29 on one A40 GPU (RunPod). Configuration: `configs/pilot.yaml`.
Code to reproduce: `bash scripts/reproduce.sh pilot`. The logs of the runs are in `logs/`.

| Model | Layer used | Step 1 | Step 2 |
|---|---|---|---|
| Qwen3-8B | 18 of 36 | done | done (3 documents) |
| gpt-oss-20b | 16 of 24 | done | done (1 document for answers) |
| Llama-3.1-8B | 7 of 32 | done | meter values valid, obey rates **not valid** |

Step 3 (the patch test in `PLAN.md`) is optional and is not done.

Llama-3.1-8B did not do the task when the document came from a tool. In 78% of these answers, it called the tool again and did not give a summary. Thus its obey rates in the tool result do not measure obedience. See "Llama-3.1-8B" in Step 2.

## Summary

1. **The meter works.** On held-out text, the probe finds the correct role for 70% (Qwen3-8B), 97% (gpt-oss-20b), and 98% (Llama-3.1-8B) of the spans. A random guess gives 20% to 25%.
2. **The role tags have more effect than the style.** In Qwen3-8B and gpt-oss-20b, the tag effect is approximately 3 times the style effect (score 0.36 and 0.34). In Llama-3.1-8B, the style has almost no effect (score 0.005).
3. **In gpt-oss-20b, the Userness predicts obedience across wrappers** (Spearman rho = 0.29, p = 0.0005). In Qwen3-8B, the model obeys almost all instructions, thus no relation is possible.
4. **The score depends on the layer in Qwen3-8B.** From layer 18 to layer 31, the score is 0.24 to 0.73. At the last four layers, it is 1.4 to 3.8. In gpt-oss-20b and Llama-3.1-8B, the score is more stable (see "Score at each layer").

## Step 1: role meter

600 C4 texts in each role, a linear probe at each layer. The eval layer is the layer with the best span accuracy on the validation texts (Qwen3-8B: see the note below).

| Model | Roles | Layer | Test accuracy (token) | Test accuracy (span) | User/tool confusion (span) |
|---|---|---|---|---|---|
| Qwen3-8B | 5 | 18 | 0.51 | **0.70** | 21% and 30% |
| gpt-oss-20b | 5 | 16 | 0.88 | **0.97** | 0% |
| Llama-3.1-8B | 4 (no thinking) | 7 | 0.91 | **0.98** | 0% |

![probe accuracy](runs/pilot/summary/probe_accuracy.png)

- All three meters pass the gate. The full confusion tables are in `runs/pilot/summary/probes.md`.
- In Qwen3-8B, user and tool are the roles that the meter confuses most. Qwen3 puts a tool result inside a user turn (`<|im_start|>user<tool_response>...`). Thus the role tags of Qwen3 put tool text near user text. gpt-oss-20b and Llama-3.1-8B have a separate role tag for tool output, and their meters do not confuse user and tool.
- **Layer choice for Qwen3-8B.** The best layer on validation is layer 2 (span accuracy 0.95). An early layer is near the token embeddings, and there the probe can read which tag token is near. Thus we used the middle layer (18), as the paper does. For Llama-3.1-8B, the selected layer (7 of 32) is also early. The same risk applies to it.

## Step 2: benchmark score

206 wrappers from the paper and 5 attack types from Liu et al. (2024), x 3 safe instructions x 2 locations (user message, tool result).
Userness is the mean P(user) on the tokens of the bare instruction. The "native" wrappers contain the special tokens of the model and are not in the score.

| Value | Qwen3-8B | gpt-oss-20b | Llama-3.1-8B |
|---|---|---|---|
| Documents for Userness / for answers | 3 / 3 | 2 / 1 | 3 / 3 |
| Tag effect (95% CI) | 0.201 (0.187 to 0.215) | 0.198 (0.178 to 0.216) | 0.976 (0.963 to 0.985) |
| Style effect (P90 minus P10, tool result) | 0.072 | 0.067 | 0.005 |
| **Score** (style effect / tag effect) | **0.36** | **0.34** | **0.005** |
| Mean Userness, tool result / user message | 0.025 / 0.225 | 0.056 / 0.254 | 0.003 / 0.979 |
| Obey rate, tool result / user message | 0.993 / 1.000 | 0.905 / 0.993 | (not valid) / 0.990 |
| Spearman, Userness vs obey rate (tool result) | rho = 0.13, p = 0.07 | **rho = 0.29, p = 0.0005** | (not valid) |

![Userness by wrapper group](runs/pilot/summary/userness_by_group.png)

### Qwen3-8B

- The role tags have the larger effect (0.20), but the style also has an effect (0.07). Wrappers that the authors labeled "user" have 6 times the Userness of the other wrappers in the tool result (0.069 against 0.010 to 0.014).
- The model obeyed 99.3% of the instructions in the tool result. Our instructions are safe and do not conflict with the task of the user. For these instructions, Qwen3-8B does not use the role to select whom to obey.
- In a real tool result, the meter reads the web page as "thinking" (0.57) more than as "tool" (0.07). The Userness values are low and comparable between conditions, but the other roles are not reliable in this model.

### gpt-oss-20b

| Group | Userness (tool) | Obey rate (tool) | Obey rate (user) |
|---|---|---|---|
| user | 0.091 | 0.98 | 1.00 |
| assistant | 0.048 | 0.88 | 0.99 |
| tool | 0.047 | 0.93 | 1.00 |
| other | 0.040 | 0.80 | 0.97 |
| attack (Liu et al.) | 0.056 | 0.73 | 1.00 |
| native | 0.143 | 0.92 | 0.94 |

- The score is almost the same as for Qwen3-8B (0.34 and 0.36).
- The meter is well calibrated in a real context. It reads the web page in a tool result as "tool" (0.68), and the request of the user as "user" (0.91).
- The obey rate is lower in the tool result (0.905) than in the user message (0.993). Thus the model uses the sender, a little, to select whom to obey.
- **The Userness predicts the obey rate.** A wrapper with more Userness in the tool result is obeyed more frequently (rho = 0.29, p = 0.0005, 199 wrappers). The "user" wrappers have the highest Userness and the highest obey rate (0.98). The attack types "context ignoring" and "combined" were obeyed in only 1 of 3 answers in the tool result.
- The summary check failed for most French answers, because it counts words that are the same in the answer and in the English document. For the other instructions, the summary was done in 99% to 100% of answers.

### Llama-3.1-8B

- **The meter values are valid.** The Userness of the instruction is 0.98 in the user message and 0.003 in the tool result. The style has almost no effect (0.005). By this meter, Llama-3.1-8B tracks the role from the role tags only. The native wrappers are the exception: they increase the Userness in the tool result to 0.088.
- In a real tool result, the meter reads the web page as "assistant" (0.89), not as "tool" (0.07). The Userness is correct, but the other roles are not reliable in this context.
- **The obey rates in the tool result are not valid.** In 1,477 of 1,899 answers (78%), Llama-3.1-8B called `fetch_url` again. Thus a low obey rate (0.18) does not show that it refused the instruction. A different tool-result format, or a system prompt, can correct this in a future run.

### Score at each layer

The score uses the eval layer. This table gives the score at other layers. The roleness file has the Userness for all layers, thus no new run was necessary.

| Model | Eval layer: score | Range of the score from the eval layer to the last layer |
|---|---|---|
| Qwen3-8B | 18: 0.36 | 0.24 to 3.80 (layers 18 to 35) |
| gpt-oss-20b | 16: 0.34 | 0.10 to 0.34 (layers 16 to 23) |
| Llama-3.1-8B | 7: 0.005 | 0.00 to 0.05 (layers 7 to 31) |

- In Llama-3.1-8B, the result is the same at all layers from layer 4 to the last layer. The style has almost no effect.
- In gpt-oss-20b, the score is 0.34 or less at all layers after the eval layer.
- In Qwen3-8B, the score is between 0.24 and 0.73 from layer 18 to layer 31. At the last four layers, the score increases to between 1.4 and 3.8. Thus the result for Qwen3-8B depends on the layer.

## Limits

1. **Llama-3.1-8B behavior.** The tool-result prompts did not work for this model (78% of the answers call the tool again). Its obey rates in the tool result are not valid.
2. **Ceiling effect in Qwen3-8B.** The obey rate is near 100% in the tool result. Thus Userness cannot be related to behavior in this model. Instructions that conflict with the task of the user (for example, the exclusive pairs of Control Illusion) will give a range of obey rates.
3. **The meter is calibrated on single turns.** In a real tool result, the meter reads the web page as "thinking" (Qwen3-8B) or "assistant" (Llama-3.1-8B). The Userness is low and comparable between conditions, but the other roles are not reliable in long contexts. gpt-oss-20b does not have this problem.
4. **Layer choice.** Qwen3-8B uses the middle layer, not the best layer on validation (layer 2). Llama-3.1-8B uses an early layer (7 of 32). The score of Qwen3-8B changes with the layer (see "Score at each layer").
5. **No causal test of the meter.** The meter reads the role from the role tags. But we did not show that the model uses this signal for its output. Thus the score is a result about the representation of the role. Step 3 in `PLAN.md` is the test for this.
6. **Small samples.** gpt-oss-20b has answers for one document in Step 2 (the configuration now gives 3, the same as the other models). The base rate has 3 to 9 answers with no instruction.
7. **Simple checks.** The French check counts stop words, and the summary check counts shared content words. The summary check is not correct for French answers.
