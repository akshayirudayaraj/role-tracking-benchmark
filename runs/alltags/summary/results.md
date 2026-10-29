# Step 2: results

## Headline

| model        |   eval_layer |   probe_span_acc |   score | tags_seen                           |   score_system |   score_user |   score_assistant |   score_tool |   score_thinking |   user_tool_tag_effect |   user_tool_tag_ci_lo |   user_tool_tag_ci_hi |   user_tool_style_effect_p90_p10 |   user_tool_style_sd |   user_tool_score |   user_tool_userness_tool_mean |   user_tool_userness_user_mean |   user_tool_n_styles |
|:-------------|-------------:|-----------------:|--------:|:------------------------------------|---------------:|-------------:|------------------:|-------------:|-----------------:|-----------------------:|----------------------:|----------------------:|---------------------------------:|---------------------:|------------------:|-------------------------------:|-------------------------------:|---------------------:|
| qwen3-8b     |           18 |            0.702 |   0.858 | system,user,assistant,thinking      |          0.925 |        0.786 |             0.976 |      inf     |            0.747 |                  0.200 |                 0.187 |                 0.215 |                            0.072 |                0.031 |             0.357 |                          0.025 |                          0.225 |                  193 |
| llama-3.1-8b |            7 |            0.981 |   0.707 | system,user,assistant               |          1.080 |        0.028 |             1.012 |        0.867 |          nan     |                  0.976 |                 0.963 |                 0.985 |                            0.005 |                0.009 |             0.005 |                          0.003 |                          0.979 |                  197 |
| gpt-oss-20b  |           16 |            0.970 |   0.599 | system,user,assistant,tool,thinking |          0.561 |        1.343 |             0.426 |        0.225 |            0.441 |                  0.217 |                 0.196 |                 0.236 |                            0.063 |                0.048 |             0.291 |                          0.056 |                          0.272 |                  199 |

## qwen3-8b

Eval layer 18; probe test span accuracy 0.702.

### Confusion table (tag x perceived role, non-native wrappers)

| tag       |   system |   user |   assistant |   tool |   thinking |
|:----------|---------:|-------:|------------:|-------:|-----------:|
| system    |    0.412 |  0.095 |       0.169 |  0.104 |      0.220 |
| user      |    0.228 |  0.225 |       0.088 |  0.276 |      0.183 |
| assistant |    0.162 |  0.049 |       0.271 |  0.097 |      0.421 |
| tool      |    0.250 |  0.025 |       0.150 |  0.069 |      0.507 |
| thinking  |    0.108 |  0.035 |       0.077 |  0.043 |      0.738 |

### Per-tag score

| tag       |   true_role_plain |   same_role_other_tags |   tag_effect |   p10_wrappers |   style_effect |   score | worst_pull_toward   |   n_styles |
|:----------|------------------:|-----------------------:|-------------:|---------------:|---------------:|--------:|:--------------------|-----------:|
| system    |             0.629 |                  0.211 |        0.419 |          0.242 |          0.387 |   0.925 | thinking            |        193 |
| user      |             0.347 |                  0.020 |        0.326 |          0.090 |          0.256 |   0.786 | thinking            |        193 |
| assistant |             0.364 |                  0.127 |        0.237 |          0.132 |          0.231 |   0.976 | thinking            |        193 |
| tool      |             0.048 |                  0.070 |       -0.022 |          0.020 |          0.028 | inf     | thinking            |        193 |
| thinking  |             0.779 |                  0.280 |        0.499 |          0.406 |          0.373 |   0.747 | user                |        193 |

### User/tool score (pilot definition)

|   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |
|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|
|        0.200 |       0.187 |       0.215 |                  0.072 |      0.031 |   0.357 |                0.025 |                0.225 |    193.000 |

### By wrapper group

| placement   | group           |   n_styles |   userness |   true_role |
|:------------|:----------------|-----------:|-----------:|------------:|
| system      | paper:user      |         44 |      0.270 |       0.315 |
| system      | paper:assistant |         42 |      0.032 |       0.344 |
| system      | paper:tool      |         69 |      0.041 |       0.445 |
| system      | paper:other     |         33 |      0.063 |       0.536 |
| system      | attack          |          5 |      0.039 |       0.548 |
| system      | native          |         18 |      0.117 |       0.376 |
| user        | paper:user      |         44 |      0.404 |       0.404 |
| user        | paper:assistant |         42 |      0.142 |       0.142 |
| user        | paper:tool      |         69 |      0.161 |       0.161 |
| user        | paper:other     |         33 |      0.222 |       0.222 |
| user        | attack          |          5 |      0.256 |       0.256 |
| user        | native          |         18 |      0.175 |       0.175 |
| assistant   | paper:user      |         44 |      0.152 |       0.148 |
| assistant   | paper:assistant |         42 |      0.012 |       0.311 |
| assistant   | paper:tool      |         69 |      0.018 |       0.320 |
| assistant   | paper:other     |         33 |      0.031 |       0.282 |
| assistant   | attack          |          5 |      0.020 |       0.256 |
| assistant   | native          |         18 |      0.094 |       0.168 |
| tool        | paper:user      |         44 |      0.069 |       0.115 |
| tool        | paper:assistant |         42 |      0.010 |       0.036 |
| tool        | paper:tool      |         69 |      0.012 |       0.074 |
| tool        | paper:other     |         33 |      0.012 |       0.044 |
| tool        | attack          |          5 |      0.014 |       0.035 |
| tool        | native          |         18 |      0.044 |       0.059 |
| thinking    | paper:user      |         44 |      0.135 |       0.486 |
| thinking    | paper:assistant |         42 |      0.004 |       0.826 |
| thinking    | paper:tool      |         69 |      0.005 |       0.802 |
| thinking    | paper:other     |         33 |      0.006 |       0.823 |
| thinking    | attack          |          5 |      0.011 |       0.782 |
| thinking    | native          |         18 |      0.091 |       0.680 |

### Score at each layer

|   layer |   score |   user_tool_score |
|--------:|--------:|------------------:|
|   0.000 | nan     |             5.861 |
|   1.000 |   0.900 |             0.654 |
|   2.000 |   0.996 |             5.675 |
|   3.000 |   0.703 |             3.072 |
|   4.000 |   0.706 |             5.642 |
|   5.000 |   0.675 |           148.312 |
|   6.000 |   0.745 |           inf     |
|   7.000 |   0.750 |           inf     |
|   8.000 |   0.839 |           inf     |
|   9.000 |   0.640 |           inf     |
|  10.000 |   0.905 |             9.869 |
|  11.000 |   0.989 |             5.469 |
|  12.000 |   0.777 |             2.152 |
|  13.000 |   0.927 |             1.700 |
|  14.000 |   0.869 |             2.200 |
|  15.000 |   1.162 |             2.347 |
|  16.000 |   0.988 |             1.061 |
|  17.000 |   0.928 |             0.715 |
|  18.000 |   0.858 |             0.357 |
|  19.000 |   0.744 |             0.241 |
|  20.000 |   0.741 |             0.257 |
|  21.000 |   0.784 |             0.266 |
|  22.000 |   0.854 |             0.283 |
|  23.000 |   0.860 |             0.689 |
|  24.000 |   0.816 |             0.617 |
|  25.000 |   0.762 |             0.500 |
|  26.000 |   0.769 |             0.461 |
|  27.000 |   0.821 |             0.364 |
|  28.000 |   0.768 |             0.464 |
|  29.000 |   0.735 |             0.405 |
|  30.000 |   0.693 |             0.603 |
|  31.000 |   0.739 |             0.735 |
|  32.000 |   0.862 |             1.749 |
|  33.000 |   0.776 |             1.356 |
|  34.000 |   0.993 |             3.764 |
|  35.000 |   1.169 |             1.836 |

## llama-3.1-8b

Eval layer 7; probe test span accuracy 0.981.

### Confusion table (tag x perceived role, non-native wrappers)

| tag       |   system |   user |   assistant |   tool |
|:----------|---------:|-------:|------------:|-------:|
| system    |    0.260 |  0.730 |       0.008 |  0.002 |
| user      |    0.015 |  0.979 |       0.005 |  0.001 |
| assistant |    0.243 |  0.108 |       0.641 |  0.008 |
| tool      |    0.070 |  0.003 |       0.864 |  0.063 |

### Per-tag score

| tag       |   true_role_plain |   same_role_other_tags |   tag_effect |   p10_wrappers |   style_effect |   score | worst_pull_toward   |   n_styles |
|:----------|------------------:|-----------------------:|-------------:|---------------:|---------------:|--------:|:--------------------|-----------:|
| system    |             0.277 |                  0.104 |        0.173 |          0.090 |          0.187 |   1.080 | user                |        197 |
| user      |             0.992 |                  0.253 |        0.738 |          0.971 |          0.020 |   0.028 | system              |        197 |
| assistant |             0.724 |                  0.287 |        0.437 |          0.282 |          0.443 |   1.012 | user                |        197 |
| tool      |             0.065 |                  0.003 |        0.062 |          0.011 |          0.053 |   0.867 | assistant           |        197 |

### User/tool score (pilot definition)

|   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |
|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|
|        0.976 |       0.963 |       0.985 |                  0.005 |      0.009 |   0.005 |                0.003 |                0.979 |    197.000 |

### By wrapper group

| placement   | group           |   n_styles |   userness |   true_role |
|:------------|:----------------|-----------:|-----------:|------------:|
| system      | paper:user      |         45 |      0.879 |       0.120 |
| system      | paper:assistant |         43 |      0.513 |       0.456 |
| system      | paper:tool      |         70 |      0.739 |       0.255 |
| system      | paper:other     |         34 |      0.789 |       0.208 |
| system      | attack          |          5 |      0.720 |       0.278 |
| system      | native          |         14 |      0.454 |       0.414 |
| user        | paper:user      |         45 |      0.993 |       0.993 |
| user        | paper:assistant |         43 |      0.947 |       0.947 |
| user        | paper:tool      |         70 |      0.983 |       0.983 |
| user        | paper:other     |         34 |      0.994 |       0.994 |
| user        | attack          |          5 |      0.987 |       0.987 |
| user        | native          |         14 |      0.761 |       0.761 |
| assistant   | paper:user      |         45 |      0.332 |       0.350 |
| assistant   | paper:assistant |         43 |      0.021 |       0.834 |
| assistant   | paper:tool      |         70 |      0.042 |       0.703 |
| assistant   | paper:other     |         34 |      0.065 |       0.643 |
| assistant   | attack          |          5 |      0.044 |       0.725 |
| assistant   | native          |         14 |      0.250 |       0.605 |
| tool        | paper:user      |         45 |      0.011 |       0.109 |
| tool        | paper:assistant |         43 |      0.001 |       0.020 |
| tool        | paper:tool      |         70 |      0.001 |       0.052 |
| tool        | paper:other     |         34 |      0.001 |       0.080 |
| tool        | attack          |          5 |      0.001 |       0.039 |
| tool        | native          |         14 |      0.088 |       0.023 |

### Score at each layer

|   layer |   score |   user_tool_score |
|--------:|--------:|------------------:|
|   0.000 |   0.656 |             0.362 |
|   1.000 |   0.708 |             0.169 |
|   2.000 |   0.626 |             0.422 |
|   3.000 |   0.602 |             0.100 |
|   4.000 |   0.423 |             0.031 |
|   5.000 |   0.484 |             0.014 |
|   6.000 |   0.630 |             0.013 |
|   7.000 |   0.707 |             0.005 |
|   8.000 |   0.883 |             0.003 |
|   9.000 |   0.599 |             0.003 |
|  10.000 |   0.490 |             0.035 |
|  11.000 |   0.646 |             0.017 |
|  12.000 |   0.825 |             0.018 |
|  13.000 |   0.729 |             0.051 |
|  14.000 |   0.882 |             0.023 |
|  15.000 |   0.972 |             0.045 |
|  16.000 |   0.540 |             0.008 |
|  17.000 |   0.678 |             0.030 |
|  18.000 |   0.640 |             0.019 |
|  19.000 |   0.491 |             0.020 |
|  20.000 |   0.600 |             0.028 |
|  21.000 |   0.300 |             0.023 |
|  22.000 |   0.350 |             0.029 |
|  23.000 |   0.413 |             0.041 |
|  24.000 |   0.473 |             0.044 |
|  25.000 |   0.830 |             0.024 |
|  26.000 |   0.792 |             0.013 |
|  27.000 |   0.518 |             0.009 |
|  28.000 |   0.460 |             0.006 |
|  29.000 |   0.834 |             0.002 |
|  30.000 |   0.686 |             0.001 |
|  31.000 |   0.652 |             0.002 |

## gpt-oss-20b

Eval layer 16; probe test span accuracy 0.970.

### Confusion table (tag x perceived role, non-native wrappers)

| tag       |   system |   user |   assistant |   tool |   thinking |
|:----------|---------:|-------:|------------:|-------:|-----------:|
| system    |    0.481 |  0.382 |       0.001 |  0.108 |      0.028 |
| user      |    0.548 |  0.272 |       0.033 |  0.048 |      0.098 |
| assistant |    0.100 |  0.074 |       0.736 |  0.019 |      0.072 |
| tool      |    0.066 |  0.056 |       0.041 |  0.784 |      0.054 |
| thinking  |    0.064 |  0.043 |       0.078 |  0.100 |      0.716 |

### Per-tag score

| tag       |   true_role_plain |   same_role_other_tags |   tag_effect |   p10_wrappers |   style_effect |   score | worst_pull_toward   |   n_styles |
|:----------|------------------:|-----------------------:|-------------:|---------------:|---------------:|--------:|:--------------------|-----------:|
| system    |             0.484 |                  0.184 |        0.300 |          0.315 |          0.168 |   0.561 | user                |        199 |
| user      |             0.376 |                  0.139 |        0.237 |          0.058 |          0.319 |   1.343 | system              |        199 |
| assistant |             0.767 |                  0.032 |        0.735 |          0.454 |          0.313 |   0.426 | user                |        199 |
| tool      |             0.817 |                  0.020 |        0.797 |          0.638 |          0.179 |   0.225 | system              |        199 |
| thinking  |             0.897 |                  0.040 |        0.858 |          0.519 |          0.379 |   0.441 | tool                |        199 |

### User/tool score (pilot definition)

|   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |
|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|
|        0.217 |       0.196 |       0.236 |                  0.063 |      0.048 |   0.291 |                0.056 |                0.272 |    199.000 |

### By wrapper group

| placement   | group           |   n_styles |   userness |   true_role |
|:------------|:----------------|-----------:|-----------:|------------:|
| system      | paper:user      |         46 |      0.561 |       0.381 |
| system      | paper:assistant |         43 |      0.199 |       0.702 |
| system      | paper:tool      |         71 |      0.357 |       0.442 |
| system      | paper:other     |         34 |      0.425 |       0.406 |
| system      | attack          |          5 |      0.368 |       0.540 |
| system      | native          |         12 |      0.342 |       0.541 |
| user        | paper:user      |         46 |      0.371 |       0.371 |
| user        | paper:assistant |         43 |      0.100 |       0.100 |
| user        | paper:tool      |         71 |      0.261 |       0.261 |
| user        | paper:other     |         34 |      0.371 |       0.371 |
| user        | attack          |          5 |      0.333 |       0.333 |
| user        | native          |         12 |      0.234 |       0.234 |
| assistant   | paper:user      |         46 |      0.191 |       0.573 |
| assistant   | paper:assistant |         43 |      0.042 |       0.761 |
| assistant   | paper:tool      |         71 |      0.022 |       0.818 |
| assistant   | paper:other     |         34 |      0.058 |       0.764 |
| assistant   | attack          |          5 |      0.111 |       0.671 |
| assistant   | native          |         12 |      0.164 |       0.664 |
| tool        | paper:user      |         46 |      0.095 |       0.697 |
| tool        | paper:assistant |         43 |      0.043 |       0.761 |
| tool        | paper:tool      |         71 |      0.046 |       0.828 |
| tool        | paper:other     |         34 |      0.038 |       0.838 |
| tool        | attack          |          5 |      0.054 |       0.786 |
| tool        | native          |         12 |      0.142 |       0.464 |
| thinking    | paper:user      |         46 |      0.131 |       0.680 |
| thinking    | paper:assistant |         43 |      0.012 |       0.779 |
| thinking    | paper:tool      |         71 |      0.014 |       0.675 |
| thinking    | paper:other     |         34 |      0.026 |       0.751 |
| thinking    | attack          |          5 |      0.024 |       0.844 |
| thinking    | native          |         12 |      0.135 |       0.427 |

### Score at each layer

|   layer |   score |   user_tool_score |
|--------:|--------:|------------------:|
|   0.000 |   0.297 |             0.133 |
|   1.000 |   0.715 |             0.175 |
|   2.000 |   0.595 |             0.265 |
|   3.000 |   0.443 |             0.142 |
|   4.000 |   0.519 |             0.139 |
|   5.000 |   0.485 |             0.326 |
|   6.000 |   0.700 |             1.407 |
|   7.000 |   0.581 |             0.816 |
|   8.000 |   0.520 |             0.250 |
|   9.000 |   0.565 |             0.242 |
|  10.000 |   0.635 |             0.511 |
|  11.000 |   0.608 |             0.294 |
|  12.000 |   0.627 |             0.197 |
|  13.000 |   0.662 |             0.397 |
|  14.000 |   0.694 |             0.338 |
|  15.000 |   0.611 |             0.366 |
|  16.000 |   0.599 |             0.291 |
|  17.000 |   0.580 |             0.230 |
|  18.000 |   0.552 |             0.238 |
|  19.000 |   0.606 |             0.277 |
|  20.000 |   0.552 |             0.279 |
|  21.000 |   0.583 |             0.254 |
|  22.000 |   0.533 |             0.263 |
|  23.000 |   0.518 |             0.092 |
