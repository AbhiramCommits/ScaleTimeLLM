| task | difficulty | model | strategy | budget | accuracy | ci_low | ci_high | mean_flops | n_problems |
|---|---|---|---|---|---|---|---|---|---|
| chartqa | easy | mock-qwen2.5-0.5b | best_of_n | 2 | 1 | 1 | 1 | 1.308e+11 | 10 |
| chartqa | hard | mock-qwen2.5-0.5b | step_search | 3 | 0.6 | 0.2 | 1 | 2.476e+11 | 5 |
| chartqa | medium | mock-qwen2.5-0.5b | weighted_vote | 4 | 0.5758 | 0.3939 | 0.7273 | 2.619e+11 | 33 |
| humaneval | easy | mock-qwen2.5-0.5b | best_of_n | 1 | 1 | 1 | 1 | 6.357e+10 | 9 |
| humaneval | hard | mock-qwen2.5-0.5b | best_of_n | 1 | 0 | 0 | 0 | 5.928e+10 | 5 |
| humaneval | medium | mock-qwen2.5-1.5b | best_of_n | 1 | 0.9091 | 0.7727 | 1 | 1.971e+11 | 22 |
| synthetic | easy | mock-qwen2.5-0.5b | best_of_n | 2 | 1 | 1 | 1 | 1.664e+11 | 16 |
| synthetic | hard | mock-qwen2.5-0.5b | step_search | 2 | 0.3077 | 0.07692 | 0.5385 | 2.422e+11 | 13 |
| synthetic | medium | mock-qwen2.5-0.5b | best_of_n | 3 | 1 | 1 | 1 | 2.572e+11 | 19 |
