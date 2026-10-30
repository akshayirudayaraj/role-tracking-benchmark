# Step 2: results

## Headline

| model        |   eval_layer |   probe_span_acc |   score | tags_seen                           |   score_system |   score_user |   score_assistant |   score_tool |   score_thinking |   user_tool_tag_effect |   user_tool_tag_ci_lo |   user_tool_tag_ci_hi |   user_tool_style_effect_p90_p10 |   user_tool_style_sd |   user_tool_score |   user_tool_userness_tool_mean |   user_tool_userness_user_mean |   user_tool_n_styles |
|:-------------|-------------:|-----------------:|--------:|:------------------------------------|---------------:|-------------:|------------------:|-------------:|-----------------:|-----------------------:|----------------------:|----------------------:|---------------------------------:|---------------------:|------------------:|-------------------------------:|-------------------------------:|---------------------:|
| qwen3-8b     |           18 |            0.702 |   0.837 | system,user,assistant,tool,thinking |          1.039 |        0.970 |             1.041 |        0.572 |            0.563 |                  0.053 |                 0.049 |                 0.057 |                            0.238 |                0.096 |             4.506 |                          0.119 |                          0.172 |                  193 |
| llama-3.1-8b |            7 |            0.981 |   0.405 | system,user,assistant,tool          |          0.450 |        0.209 |             0.381 |        0.579 |          nan     |                  0.597 |                 0.578 |                 0.612 |                            0.363 |                0.135 |             0.608 |                          0.272 |                          0.869 |                  197 |
| gpt-oss-20b  |           16 |            0.970 |   0.421 | system,user,assistant,tool,thinking |          0.328 |        0.768 |             0.355 |        0.152 |            0.503 |                  0.500 |                 0.473 |                 0.526 |                            0.050 |                0.027 |             0.099 |                          0.015 |                          0.515 |                  199 |

## qwen3-8b

Eval layer 18; probe test span accuracy 0.702.

### Confusion table (tag x perceived role, non-native wrappers)

| tag       |   system |   user |   assistant |   tool |   thinking |
|:----------|---------:|-------:|------------:|-------:|-----------:|
| system    |    0.478 |  0.062 |       0.186 |  0.080 |      0.193 |
| user      |    0.336 |  0.172 |       0.152 |  0.228 |      0.112 |
| assistant |    0.302 |  0.029 |       0.275 |  0.035 |      0.359 |
| tool      |    0.314 |  0.119 |       0.165 |  0.248 |      0.155 |
| thinking  |    0.183 |  0.018 |       0.142 |  0.024 |      0.634 |

### Per-tag score

| tag       |   true_role_plain |   same_role_other_tags |   tag_effect |   p10_wrappers |   style_effect |   score | worst_pull_toward   |   n_styles |
|:----------|------------------:|-----------------------:|-------------:|---------------:|---------------:|--------:|:--------------------|-----------:|
| system    |             0.712 |                  0.346 |        0.366 |          0.332 |          0.380 |   1.039 | thinking            |        193 |
| user      |             0.260 |                  0.043 |        0.216 |          0.050 |          0.210 |   0.970 | system              |        193 |
| assistant |             0.388 |                  0.137 |        0.251 |          0.126 |          0.262 |   1.041 | system              |        193 |
| tool      |             0.210 |                  0.039 |        0.171 |          0.112 |          0.098 |   0.572 | system              |        193 |
| thinking  |             0.653 |                  0.130 |        0.523 |          0.358 |          0.295 |   0.563 | system              |        193 |

### User/tool score (pilot definition)

|   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |
|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|
|        0.053 |       0.049 |       0.057 |                  0.238 |      0.096 |   4.506 |                0.119 |                0.172 |    193.000 |

### By wrapper group

| placement   | group           |   n_styles |   userness |   true_role |
|:------------|:----------------|-----------:|-----------:|------------:|
| system      | paper:user      |         44 |      0.212 |       0.408 |
| system      | paper:assistant |         42 |      0.016 |       0.441 |
| system      | paper:tool      |         69 |      0.018 |       0.481 |
| system      | paper:other     |         33 |      0.022 |       0.587 |
| system      | attack          |          5 |      0.008 |       0.638 |
| system      | native          |         18 |      0.076 |       0.425 |
| user        | paper:user      |         44 |      0.324 |       0.324 |
| user        | paper:assistant |         42 |      0.105 |       0.105 |
| user        | paper:tool      |         69 |      0.116 |       0.116 |
| user        | paper:other     |         33 |      0.171 |       0.171 |
| user        | attack          |          5 |      0.165 |       0.165 |
| user        | native          |         18 |      0.123 |       0.123 |
| assistant   | paper:user      |         44 |      0.115 |       0.143 |
| assistant   | paper:assistant |         42 |      0.002 |       0.327 |
| assistant   | paper:tool      |         69 |      0.003 |       0.326 |
| assistant   | paper:other     |         33 |      0.004 |       0.282 |
| assistant   | attack          |          5 |      0.001 |       0.251 |
| assistant   | native          |         18 |      0.051 |       0.200 |
| tool        | paper:user      |         44 |      0.257 |       0.306 |
| tool        | paper:assistant |         42 |      0.066 |       0.169 |
| tool        | paper:tool      |         69 |      0.071 |       0.270 |
| tool        | paper:other     |         33 |      0.105 |       0.238 |
| tool        | attack          |          5 |      0.098 |       0.150 |
| tool        | native          |         18 |      0.093 |       0.156 |
| thinking    | paper:user      |         44 |      0.075 |       0.417 |
| thinking    | paper:assistant |         42 |      0.001 |       0.690 |
| thinking    | paper:tool      |         69 |      0.001 |       0.696 |
| thinking    | paper:other     |         33 |      0.001 |       0.706 |
| thinking    | attack          |          5 |      0.000 |       0.738 |
| thinking    | native          |         18 |      0.039 |       0.629 |

### Score at each layer

|   layer |   score |   user_tool_score |
|--------:|--------:|------------------:|
|   0.000 | nan     |             6.805 |
|   1.000 |   1.311 |             6.144 |
|   2.000 |   0.807 |             4.431 |
|   3.000 |   0.635 |             2.625 |
|   4.000 |   0.679 |             5.015 |
|   5.000 |   0.640 |             4.441 |
|   6.000 |   0.724 |             9.280 |
|   7.000 |   0.738 |             8.231 |
|   8.000 |   0.810 |             8.707 |
|   9.000 |   0.844 |             8.482 |
|  10.000 |   0.831 |             7.309 |
|  11.000 |   0.814 |             8.045 |
|  12.000 |   0.514 |             4.478 |
|  13.000 |   0.590 |             5.024 |
|  14.000 |   0.599 |             6.210 |
|  15.000 |   0.803 |             6.316 |
|  16.000 |   0.733 |             4.452 |
|  17.000 |   0.744 |             4.241 |
|  18.000 |   0.837 |             4.506 |
|  19.000 |   0.811 |             3.187 |
|  20.000 |   0.844 |             4.174 |
|  21.000 |   0.861 |             4.225 |
|  22.000 |   0.817 |             4.063 |
|  23.000 |   0.853 |             6.117 |
|  24.000 |   0.810 |             4.822 |
|  25.000 |   0.844 |             4.698 |
|  26.000 |   0.920 |             4.106 |
|  27.000 |   0.872 |             4.018 |
|  28.000 |   0.924 |             4.819 |
|  29.000 |   0.883 |             4.491 |
|  30.000 |   0.878 |             3.800 |
|  31.000 |   0.967 |             4.561 |
|  32.000 |   1.014 |             6.043 |
|  33.000 |   0.983 |             6.262 |
|  34.000 |   0.997 |             4.438 |
|  35.000 |   0.802 |             2.402 |

## llama-3.1-8b

Eval layer 7; probe test span accuracy 0.981.

### Confusion table (tag x perceived role, non-native wrappers)

| tag       |   system |   user |   assistant |   tool |
|:----------|---------:|-------:|------------:|-------:|
| system    |    0.717 |  0.173 |       0.041 |  0.068 |
| user      |    0.104 |  0.869 |       0.007 |  0.020 |
| assistant |    0.284 |  0.018 |       0.680 |  0.018 |
| tool      |    0.239 |  0.272 |       0.007 |  0.482 |

### Per-tag score

| tag       |   true_role_plain |   same_role_other_tags |   tag_effect |   p10_wrappers |   style_effect |   score | worst_pull_toward   |   n_styles |
|:----------|------------------:|-----------------------:|-------------:|---------------:|---------------:|--------:|:--------------------|-----------:|
| system    |             0.814 |                  0.163 |        0.651 |          0.521 |          0.293 |   0.450 | user                |        197 |
| user      |             0.897 |                  0.110 |        0.787 |          0.732 |          0.165 |   0.209 | system              |        197 |
| assistant |             0.731 |                  0.004 |        0.726 |          0.454 |          0.277 |   0.381 | system              |        197 |
| tool      |             0.643 |                  0.028 |        0.614 |          0.287 |          0.355 |   0.579 | user                |        197 |

### User/tool score (pilot definition)

|   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |
|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|
|        0.597 |       0.578 |       0.612 |                  0.363 |      0.135 |   0.608 |                0.272 |                0.869 |    197.000 |

### By wrapper group

| placement   | group           |   n_styles |   userness |   true_role |
|:------------|:----------------|-----------:|-----------:|------------:|
| system      | paper:user      |         45 |      0.351 |       0.574 |
| system      | paper:assistant |         43 |      0.088 |       0.736 |
| system      | paper:tool      |         70 |      0.123 |       0.783 |
| system      | paper:other     |         34 |      0.157 |       0.736 |
| system      | attack          |          5 |      0.124 |       0.788 |
| system      | native          |         14 |      0.180 |       0.586 |
| user        | paper:user      |         45 |      0.937 |       0.937 |
| user        | paper:assistant |         43 |      0.729 |       0.729 |
| user        | paper:tool      |         70 |      0.886 |       0.886 |
| user        | paper:other     |         34 |      0.921 |       0.921 |
| user        | attack          |          5 |      0.862 |       0.862 |
| user        | native          |         14 |      0.590 |       0.590 |
| assistant   | paper:user      |         45 |      0.055 |       0.475 |
| assistant   | paper:assistant |         43 |      0.004 |       0.838 |
| assistant   | paper:tool      |         70 |      0.006 |       0.723 |
| assistant   | paper:other     |         34 |      0.010 |       0.651 |
| assistant   | attack          |          5 |      0.005 |       0.769 |
| assistant   | native          |         14 |      0.076 |       0.694 |
| tool        | paper:user      |         45 |      0.443 |       0.374 |
| tool        | paper:assistant |         43 |      0.177 |       0.463 |
| tool        | paper:tool      |         70 |      0.225 |       0.528 |
| tool        | paper:other     |         34 |      0.259 |       0.555 |
| tool        | attack          |          5 |      0.299 |       0.464 |
| tool        | native          |         14 |      0.331 |       0.200 |

### Score at each layer

|   layer |   score |   user_tool_score |
|--------:|--------:|------------------:|
|   0.000 |   0.694 |             0.521 |
|   1.000 |   0.593 |             0.118 |
|   2.000 |   0.623 |             0.320 |
|   3.000 |   0.590 |             0.198 |
|   4.000 |   0.443 |             0.178 |
|   5.000 |   0.458 |             0.217 |
|   6.000 |   0.499 |             0.431 |
|   7.000 |   0.405 |             0.608 |
|   8.000 |   0.457 |             0.458 |
|   9.000 |   0.418 |             0.640 |
|  10.000 |   0.441 |             0.727 |
|  11.000 |   0.352 |             0.317 |
|  12.000 |   0.371 |             0.444 |
|  13.000 |   0.401 |             0.487 |
|  14.000 |   0.403 |             0.607 |
|  15.000 |   0.427 |             0.651 |
|  16.000 |   0.500 |             0.552 |
|  17.000 |   0.494 |             0.664 |
|  18.000 |   0.515 |             0.483 |
|  19.000 |   0.534 |             0.583 |
|  20.000 |   0.509 |             0.597 |
|  21.000 |   0.491 |             0.586 |
|  22.000 |   0.481 |             0.525 |
|  23.000 |   0.510 |             0.630 |
|  24.000 |   0.526 |             0.474 |
|  25.000 |   0.534 |             0.318 |
|  26.000 |   0.458 |             0.171 |
|  27.000 |   0.425 |             0.109 |
|  28.000 |   0.437 |             0.085 |
|  29.000 |   0.428 |             0.052 |
|  30.000 |   0.429 |             0.015 |
|  31.000 |   0.438 |             0.012 |

## gpt-oss-20b

Eval layer 16; probe test span accuracy 0.970.

### Confusion table (tag x perceived role, non-native wrappers)

| tag       |   system |   user |   assistant |   tool |   thinking |
|:----------|---------:|-------:|------------:|-------:|-----------:|
| system    |    0.778 |  0.163 |       0.005 |  0.040 |      0.013 |
| user      |    0.412 |  0.515 |       0.007 |  0.038 |      0.027 |
| assistant |    0.152 |  0.037 |       0.731 |  0.019 |      0.062 |
| tool      |    0.064 |  0.015 |       0.000 |  0.913 |      0.008 |
| thinking  |    0.256 |  0.071 |       0.023 |  0.103 |      0.548 |

### Per-tag score

| tag       |   true_role_plain |   same_role_other_tags |   tag_effect |   p10_wrappers |   style_effect |   score | worst_pull_toward   |   n_styles |
|:----------|------------------:|-----------------------:|-------------:|---------------:|---------------:|--------:|:--------------------|-----------:|
| system    |             0.782 |                  0.161 |        0.621 |          0.578 |          0.204 |   0.328 | user                |        199 |
| user      |             0.701 |                  0.081 |        0.619 |          0.225 |          0.476 |   0.768 | system              |        199 |
| assistant |             0.728 |                  0.003 |        0.725 |          0.471 |          0.258 |   0.355 | system              |        199 |
| tool      |             0.942 |                  0.009 |        0.933 |          0.800 |          0.142 |   0.152 | system              |        199 |
| thinking  |             0.726 |                  0.027 |        0.699 |          0.374 |          0.351 |   0.503 | system              |        199 |

### User/tool score (pilot definition)

|   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |
|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|
|        0.500 |       0.473 |       0.526 |                  0.050 |      0.027 |   0.099 |                0.015 |                0.515 |    199.000 |

### By wrapper group

| placement   | group           |   n_styles |   userness |   true_role |
|:------------|:----------------|-----------:|-----------:|------------:|
| system      | paper:user      |         46 |      0.366 |       0.605 |
| system      | paper:assistant |         43 |      0.067 |       0.893 |
| system      | paper:tool      |         71 |      0.090 |       0.831 |
| system      | paper:other     |         34 |      0.162 |       0.752 |
| system      | attack          |          5 |      0.177 |       0.798 |
| system      | native          |         12 |      0.181 |       0.758 |
| user        | paper:user      |         46 |      0.682 |       0.682 |
| user        | paper:assistant |         43 |      0.277 |       0.277 |
| user        | paper:tool      |         71 |      0.485 |       0.485 |
| user        | paper:other     |         34 |      0.642 |       0.642 |
| user        | attack          |          5 |      0.591 |       0.591 |
| user        | native          |         12 |      0.392 |       0.392 |
| assistant   | paper:user      |         46 |      0.130 |       0.530 |
| assistant   | paper:assistant |         43 |      0.012 |       0.747 |
| assistant   | paper:tool      |         71 |      0.003 |       0.834 |
| assistant   | paper:other     |         34 |      0.016 |       0.771 |
| assistant   | attack          |          5 |      0.035 |       0.692 |
| assistant   | native          |         12 |      0.079 |       0.666 |
| tool        | paper:user      |         46 |      0.052 |       0.819 |
| tool        | paper:assistant |         43 |      0.005 |       0.908 |
| tool        | paper:tool      |         71 |      0.002 |       0.952 |
| tool        | paper:other     |         34 |      0.004 |       0.961 |
| tool        | attack          |          5 |      0.006 |       0.932 |
| tool        | native          |         12 |      0.070 |       0.636 |
| thinking    | paper:user      |         46 |      0.196 |       0.462 |
| thinking    | paper:assistant |         43 |      0.027 |       0.605 |
| thinking    | paper:tool      |         71 |      0.026 |       0.527 |
| thinking    | paper:other     |         34 |      0.048 |       0.621 |
| thinking    | attack          |          5 |      0.070 |       0.657 |
| thinking    | native          |         12 |      0.132 |       0.333 |

### Score at each layer

|   layer |   score |   user_tool_score |
|--------:|--------:|------------------:|
|   0.000 |   0.344 |             0.143 |
|   1.000 |   0.618 |             0.209 |
|   2.000 |   0.679 |             0.392 |
|   3.000 |   0.539 |             0.094 |
|   4.000 |   0.570 |             0.165 |
|   5.000 |   0.560 |             0.254 |
|   6.000 |   0.650 |             0.576 |
|   7.000 |   0.539 |             0.177 |
|   8.000 |   0.468 |             0.094 |
|   9.000 |   0.398 |             0.060 |
|  10.000 |   0.494 |             0.233 |
|  11.000 |   0.461 |             0.168 |
|  12.000 |   0.471 |             0.124 |
|  13.000 |   0.440 |             0.254 |
|  14.000 |   0.445 |             0.105 |
|  15.000 |   0.467 |             0.172 |
|  16.000 |   0.421 |             0.099 |
|  17.000 |   0.397 |             0.116 |
|  18.000 |   0.414 |             0.048 |
|  19.000 |   0.419 |             0.117 |
|  20.000 |   0.459 |             0.242 |
|  21.000 |   0.529 |             0.283 |
|  22.000 |   0.528 |             0.298 |
|  23.000 |   0.544 |             0.251 |
