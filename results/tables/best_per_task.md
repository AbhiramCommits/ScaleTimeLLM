| task | model | strategy | budget | accuracy | ci_low | ci_high | tokens_per_correct | dollars_per_correct | mean_flops |
|---|---|---|---|---|---|---|---|---|---|
| chartqa | mock-qwen2.5-1.5b | self_consistency | 8 | 0.9375 | 0.8542 | 1 | 565.3 | 0.000127 | 1.632e+12 |
| humaneval | mock-qwen2.5-1.5b | self_consistency | 4 | 0.8333 | 0.7292 | 0.9375 | 297.1 | 9.678e-05 | 7.626e+11 |
| synthetic | mock-qwen2.5-1.5b | weighted_vote | 8 | 1 | 1 | 1 | 695.5 | 0.0002688 | 2.142e+12 |
