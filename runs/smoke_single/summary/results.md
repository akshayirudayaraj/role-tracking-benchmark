# Step 2: results

## Headline

| model      |   eval_layer |   probe_span_acc |   score | tags_seen     |   score_system |   score_user |   score_assistant |   score_tool |   score_thinking |   user_tool_tag_effect |   user_tool_tag_ci_lo |   user_tool_tag_ci_hi |   user_tool_style_effect_p90_p10 |   user_tool_style_sd |   user_tool_score |   user_tool_userness_tool_mean |   user_tool_userness_user_mean |   user_tool_n_styles |
|:-----------|-------------:|-----------------:|--------:|:--------------|---------------:|-------------:|------------------:|-------------:|-----------------:|-----------------------:|----------------------:|----------------------:|---------------------------------:|---------------------:|------------------:|-------------------------------:|-------------------------------:|---------------------:|
| qwen3-0.6b |            7 |            0.800 |   0.747 | user,thinking |         -0.138 |        0.954 |             0.979 |        1.155 |            0.539 |                  0.016 |                 0.008 |                 0.026 |                            0.095 |                0.052 |             5.784 |                          0.177 |                          0.193 |                    4 |

## qwen3-0.6b

Eval layer 7; probe test span accuracy 0.800.

### Confusion table (tag x perceived role, non-native wrappers)

| tag       |   system |   user |   assistant |   tool |   thinking |
|:----------|---------:|-------:|------------:|-------:|-----------:|
| system    |    0.407 |  0.114 |       0.176 |  0.118 |      0.184 |
| user      |    0.351 |  0.193 |       0.146 |  0.163 |      0.148 |
| assistant |    0.367 |  0.114 |       0.191 |  0.120 |      0.208 |
| tool      |    0.347 |  0.177 |       0.146 |  0.169 |      0.162 |
| thinking  |    0.354 |  0.097 |       0.176 |  0.117 |      0.257 |

### Per-tag score

| tag       |   true_role_plain |   same_role_other_tags |   tag_effect |   p10_wrappers |   style_effect |   score | worst_pull_toward   |   n_styles |
|:----------|------------------:|-----------------------:|-------------:|---------------:|---------------:|--------:|:--------------------|-----------:|
| system    |             0.351 |                  0.283 |        0.069 |          0.361 |         -0.010 |  -0.138 | thinking            |          4 |
| user      |             0.236 |                  0.135 |        0.101 |          0.140 |          0.097 |   0.954 | system              |          4 |
| assistant |             0.219 |                  0.172 |        0.047 |          0.173 |          0.046 |   0.979 | system              |          4 |
| tool      |             0.203 |                  0.147 |        0.057 |          0.138 |          0.065 |   1.155 | system              |          4 |
| thinking  |             0.291 |                  0.189 |        0.102 |          0.236 |          0.055 |   0.539 | system              |          4 |

### User/tool score (pilot definition)

|   tag_effect |   tag_ci_lo |   tag_ci_hi |   style_effect_p90_p10 |   style_sd |   score |   userness_tool_mean |   userness_user_mean |   n_styles |
|-------------:|------------:|------------:|-----------------------:|-----------:|--------:|---------------------:|---------------------:|-----------:|
|        0.016 |       0.008 |       0.026 |                  0.095 |      0.052 |   5.784 |                0.177 |                0.193 |      4.000 |

### By wrapper group

| placement   | group      |   n_styles |   userness |   true_role |
|:------------|:-----------|-----------:|-----------:|------------:|
| system      | paper:user |          2 |      0.129 |       0.385 |
| system      | attack     |          2 |      0.099 |       0.430 |
| system      | native     |          2 |      0.194 |       0.321 |
| user        | paper:user |          2 |      0.207 |       0.207 |
| user        | attack     |          2 |      0.179 |       0.179 |
| user        | native     |          2 |      0.241 |       0.241 |
| assistant   | paper:user |          2 |      0.133 |       0.188 |
| assistant   | attack     |          2 |      0.095 |       0.194 |
| assistant   | native     |          2 |      0.193 |       0.155 |
| tool        | paper:user |          2 |      0.196 |       0.174 |
| tool        | attack     |          2 |      0.157 |       0.165 |
| tool        | native     |          2 |      0.223 |       0.193 |
| thinking    | paper:user |          2 |      0.114 |       0.247 |
| thinking    | attack     |          2 |      0.079 |       0.266 |
| thinking    | native     |          2 |      0.172 |       0.244 |

### Score at each layer

|   layer |   score |   user_tool_score |
|--------:|--------:|------------------:|
|   0.000 | nan     |             2.047 |
|   7.000 |   0.747 |             5.784 |
|  14.000 |   0.732 |             4.486 |
|  21.000 |   0.571 |             3.034 |
