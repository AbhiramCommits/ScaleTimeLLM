| task | small_model | small_strategy | small_budget | small_accuracy | small_flops | big_model | big_strategy | big_accuracy | big_flops | equal_flops_ratio | delta_accuracy | delta_ci_low | delta_ci_high | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chartqa | mock-qwen2.5-0.5b | self_consistency | 3 | 0.5417 | 1.964e+11 | mock-qwen2.5-1.5b | greedy | 0.5625 | 2.04e+11 | 0.9624 | -0.02083 | -0.2083 | 0.1667 | statistical_tie |
| humaneval | mock-qwen2.5-0.5b | best_of_n | 3 | 0.5833 | 1.821e+11 | mock-qwen2.5-1.5b | greedy | 0.6667 | 1.916e+11 | 0.9506 | -0.08333 | -0.2708 | 0.125 | statistical_tie |
| synthetic | mock-qwen2.5-0.5b | best_of_n | 3 | 0.7292 | 2.582e+11 | mock-qwen2.5-1.5b | greedy | 0.6042 | 2.676e+11 | 0.9647 | 0.125 | -0.04167 | 0.2917 | statistical_tie |
