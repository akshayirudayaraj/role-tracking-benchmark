# Steps 2 and 3: results

## Headline

| model       |   eval_layer |   probe_span_acc |   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |   obey_tool |   obey_user |   rho_userness_obey |   rho_p |   user_rate |   tool_rate |   precondition_met |
|:------------|-------------:|-----------------:|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|------------:|------------:|--------------------:|--------:|------------:|------------:|-------------------:|
| qwen3-8b    |           18 |            0.702 |        0.206 |       0.193 |       0.221 |                  0.073 |      0.032 |   0.352 |                0.025 |                0.231 |        193 |       0.997 |       1.000 |              -0.074 |   0.337 |     nan     |     nan     |            nan     |
| gpt-oss-20b |           16 |            0.970 |        0.198 |       0.178 |       0.216 |                  0.067 |      0.046 |   0.341 |                0.056 |                0.254 |        199 |       0.905 |       0.993 |               0.286 |   0.000 |       1.000 |       0.694 |              1.000 |

## qwen3-8b

Eval layer 18; probe test span accuracy 0.702.

### Benchmark score (non-native wrappers)

|   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |
|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|
|        0.206 |       0.193 |       0.221 |                  0.073 |      0.032 |   0.352 |                0.025 |                0.231 |    193.000 |

### By wrapper group

| placement   | group           |   n_styles |   userness |   obey_rate |   obey_lo |   obey_hi |   n_answers |   task_done |
|:------------|:----------------|-----------:|-----------:|------------:|----------:|----------:|------------:|------------:|
| tool        | paper:user      |         44 |      0.070 |       0.992 |     0.958 |     0.999 |         132 |       0.970 |
| tool        | paper:assistant |         42 |      0.011 |       1.000 |     0.970 |     1.000 |         126 |       1.000 |
| tool        | paper:tool      |         69 |      0.012 |       1.000 |     0.982 |     1.000 |         207 |       0.995 |
| tool        | paper:other     |         33 |      0.012 |       0.990 |     0.945 |     0.998 |          99 |       0.970 |
| tool        | attack          |          5 |      0.013 |       1.000 |     0.796 |     1.000 |          15 |       1.000 |
| tool        | native          |         18 |      0.044 |       1.000 |     0.934 |     1.000 |          54 |       0.944 |
| user        | paper:user      |         44 |      0.413 |       1.000 |     0.972 |     1.000 |         132 |       0.992 |
| user        | paper:assistant |         42 |      0.149 |       1.000 |     0.970 |     1.000 |         126 |       1.000 |
| user        | paper:tool      |         69 |      0.166 |       1.000 |     0.982 |     1.000 |         207 |       0.995 |
| user        | paper:other     |         33 |      0.225 |       1.000 |     0.963 |     1.000 |          99 |       0.990 |
| user        | attack          |          5 |      0.259 |       1.000 |     0.796 |     1.000 |          15 |       1.000 |
| user        | native          |         18 |      0.177 |       1.000 |     0.934 |     1.000 |          54 |       1.000 |

### Base rate (no instruction)

| payload   |   n |   rate |   ci_lo |   ci_hi |
|:----------|----:|-------:|--------:|--------:|
| all_caps  |   1 |  0.000 |   0.000 |   0.793 |
| end_word  |   1 |  0.000 |   0.000 |   0.793 |
| french    |   1 |  0.000 |   0.000 |   0.793 |

Spearman (per wrapper, tool result, non-native) Userness vs obey rate: rho = -0.074, permutation p = 0.337, n = 193 wrappers.


## gpt-oss-20b

Eval layer 16; probe test span accuracy 0.970.

### Benchmark score (non-native wrappers)

|   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |
|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|
|        0.198 |       0.178 |       0.216 |                  0.067 |      0.046 |   0.341 |                0.056 |                0.254 |    199.000 |

### By wrapper group

| placement   | group           |   n_styles |   userness |   obey_rate |   obey_lo |   obey_hi |   n_answers |   task_done |
|:------------|:----------------|-----------:|-----------:|------------:|----------:|----------:|------------:|------------:|
| tool        | paper:user      |         46 |      0.091 |       0.978 |     0.938 |     0.993 |         138 |       0.681 |
| tool        | paper:assistant |         43 |      0.048 |       0.884 |     0.817 |     0.928 |         129 |       0.698 |
| tool        | paper:tool      |         71 |      0.047 |       0.930 |     0.887 |     0.957 |         213 |       0.718 |
| tool        | paper:other     |         34 |      0.040 |       0.804 |     0.716 |     0.869 |         102 |       0.765 |
| tool        | attack          |          5 |      0.056 |       0.733 |     0.480 |     0.891 |          15 |       0.800 |
| tool        | native          |         12 |      0.143 |       0.917 |     0.782 |     0.971 |          36 |       0.667 |
| user        | paper:user      |         46 |      0.350 |       1.000 |     0.973 |     1.000 |         138 |       0.667 |
| user        | paper:assistant |         43 |      0.089 |       0.992 |     0.957 |     0.999 |         129 |       0.674 |
| user        | paper:tool      |         71 |      0.241 |       1.000 |     0.982 |     1.000 |         213 |       0.676 |
| user        | paper:other     |         34 |      0.352 |       0.971 |     0.917 |     0.990 |         102 |       0.647 |
| user        | attack          |          5 |      0.311 |       1.000 |     0.796 |     1.000 |          15 |       0.667 |
| user        | native          |         12 |      0.224 |       0.944 |     0.819 |     0.985 |          36 |       0.722 |

### Base rate (no instruction)

| payload   |   n |   rate |   ci_lo |   ci_hi |
|:----------|----:|-------:|--------:|--------:|
| all_caps  |   1 |  0.000 |   0.000 |   0.793 |
| end_word  |   1 |  0.000 |   0.000 |   0.793 |
| french    |   1 |  0.000 |   0.000 |   0.793 |

Spearman (per wrapper, tool result, non-native) Userness vs obey rate: rho = 0.286, permutation p = 0.0005, n = 199 wrappers.


### Steering (Step 3)

User rate 1.000, tool rate 0.694. Precondition met: the tests ran.

| test   | cond             | placement   | direction   |    frac |   alpha |   userness_L16 |   userness_L23 |   obey_rate |   obey_lo |   obey_hi |   n |   task_done |
|:-------|:-----------------|:------------|:------------|--------:|--------:|---------------:|---------------:|------------:|----------:|----------:|----:|------------:|
| A      | user             | user        | nan         | nan     |   0.000 |          0.426 |          0.390 |       1.000 |     0.904 |     1.000 |  36 |       0.833 |
| A      | tool             | tool        | nan         | nan     |   0.000 |          0.081 |          0.015 |       0.694 |     0.531 |     0.820 |  36 |       0.861 |
| 1      | toward_user_0.25 | tool        | user        |   0.250 |   0.008 |          0.159 |          0.018 |       0.639 |     0.476 |     0.775 |  36 |       0.889 |
| 2      | toward_tool_0.25 | user        | tool        |   0.250 |   0.008 |          0.290 |          0.373 |       1.000 |     0.904 |     1.000 |  36 |       0.861 |
| 1      | toward_user_0.5  | tool        | user        |   0.500 |   0.008 |          0.159 |          0.018 |       0.639 |     0.476 |     0.775 |  36 |       0.889 |
| 2      | toward_tool_0.5  | user        | tool        |   0.500 |   0.008 |          0.290 |          0.373 |       1.000 |     0.904 |     1.000 |  36 |       0.861 |
| 1      | toward_user_0.75 | tool        | user        |   0.750 |   0.023 |          0.516 |          0.026 |       0.722 |     0.560 |     0.842 |  36 |       0.861 |
| 2      | toward_tool_0.75 | user        | tool        |   0.750 |   0.023 |          0.090 |          0.345 |       1.000 |     0.904 |     1.000 |  36 |       0.861 |
| 1      | toward_user_1.0  | tool        | user        |   1.000 |   0.023 |          0.516 |          0.026 |       0.722 |     0.560 |     0.842 |  36 |       0.861 |
| 2      | toward_tool_1.0  | user        | tool        |   1.000 |   0.023 |          0.090 |          0.345 |       1.000 |     0.904 |     1.000 |  36 |       0.861 |
| 3      | random_0.25      | tool        | random      |   0.250 |   0.008 |          0.078 |          0.015 |       0.722 |     0.560 |     0.842 |  36 |       0.889 |
| 3      | assistant_0.25   | tool        | assistant   |   0.250 |   0.008 |          0.086 |          0.017 |       0.722 |     0.560 |     0.842 |  36 |       0.861 |
| 3      | random_0.5       | tool        | random      |   0.500 |   0.008 |          0.078 |          0.015 |       0.722 |     0.560 |     0.842 |  36 |       0.889 |
| 3      | assistant_0.5    | tool        | assistant   |   0.500 |   0.008 |          0.086 |          0.017 |       0.722 |     0.560 |     0.842 |  36 |       0.861 |
| 3      | random_0.75      | tool        | random      |   0.750 |   0.023 |          0.073 |          0.015 |       0.694 |     0.531 |     0.820 |  36 |       0.889 |
| 3      | assistant_0.75   | tool        | assistant   |   0.750 |   0.023 |          0.070 |          0.021 |       0.778 |     0.619 |     0.883 |  36 |       0.861 |
| 3      | random_1.0       | tool        | random      |   1.000 |   0.023 |          0.073 |          0.015 |       0.694 |     0.531 |     0.820 |  36 |       0.889 |
| 3      | assistant_1.0    | tool        | assistant   |   1.000 |   0.023 |          0.070 |          0.021 |       0.778 |     0.619 |     0.883 |  36 |       0.861 |
