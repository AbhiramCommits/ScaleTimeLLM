# Inference-Time Scaling for Small Open Language Models: A Reproducible CPU Study

**ScaleTimeLLM contributors**

## Abstract

We study inference-time scaling for reasoning and code generation with small open language models. The
repository compares greedy decoding, self-consistency, best-of-N, verifier-weighted voting and
step-level beam search over a grid of sampling budgets, two model sizes, three task families and three
seeds. Every run records token counts, estimated FLOPs, wall time and dollar cost, and the analysis
reports bootstrap confidence intervals and compute-optimal frontiers. The committed full sweep
comprises 666 configurations and 10,656 problem-seed evaluations and runs in 251.49 seconds on an
8-CPU Apple laptop. Because a full pretrained-model sweep at this grid size is infeasible on the
available CPU, the sweep uses a deterministic synthetic backend (`MockLM`) with controlled per-sample
accuracy; the serving section separately measures real `Qwen/Qwen2.5-0.5B-Instruct` weights. In this
build, verifier-weighted voting is the strongest synthetic-math strategy (accuracy 1.000 at N=8), while
self-consistency is strongest on code (0.833 at N=4) and chart QA (0.938 at N=8). At matched FLOPs,
small-model sampling versus larger-model greedy is a statistical tie on all three tasks, with point
estimates of +12.5, -8.3 and -2.1 percentage points for math, code and chart QA respectively. The
text-winning strategy does not transfer: on chart QA it reaches 0.667 versus 0.802 for
self-consistency. The verifier reaches AUROC 0.9952 and ECE 0.0160 in-distribution but leaves a 0.354
to 0.729 accuracy gap to an oracle on code and charts. We release all configs, raw JSONL, seeds,
figures, tables and reproduction commands.

## 1. Introduction

Scaling model parameters and scaling test-time computation are two ways to buy accuracy. For small
open models, test-time methods such as majority voting, reranking and search are attractive because
they can be applied without retraining the base model. The practical question is not only whether a
strategy helps, but where it sits on the accuracy-versus-compute frontier and whether the answer
transfers across task families.

This study makes four contributions:

1. A config-driven harness that measures accuracy, exact token use, estimated FLOPs, latency, dollars
   per correct answer, and strategy traces for a cross product of models, tasks, strategies, budgets
   and seeds.
2. Cached difficulty buckets derived from a cheap k=4 pilot pass rate, and compute-optimal selection
   within each task and bucket.
3. A verifier pipeline with outcome BCE, a custom pairwise margin objective, a TRL
   `RewardTrainer` pairwise variant, and held-out AUROC/ECE/top-1 evaluation.
4. A multimodal transfer test on a chart/diagram QA interface, plus a real HF serving benchmark.

The main sweep is deliberately labelled synthetic. It is a controlled study of the experimental
machinery and of strategy behaviour under known correctness rates, not a claim about Qwen accuracy.

## 2. Setup

### 2.1 Backend and models

`MockLM` is a deterministic synthetic language-model backend. It parses a synthetic task prompt,
computes the reference answer, and draws a seeded Bernoulli correctness indicator with a configurable
probability. Correct samples emit a short chain-of-thought and the correct answer; incorrect samples
emit a plausible distractor. Token counts use a deterministic whitespace/punctuation tokenizer, and
verbosity controls completion length.

The two committed model stand-ins are `mock-qwen2.5-0.5b` (494,032,768 parameters, base accuracy 0.42)
and `mock-qwen2.5-1.5b` (1,540,000,000 parameters, base accuracy 0.56). The parameter counts enter the
FLOPs formula, so the larger model is genuinely more expensive per token. Synthetic arithmetic prompts
also carry a difficulty label that shifts correctness by +/-0.28; code and chart prompts use the base
rate. The real serving check uses `Qwen/Qwen2.5-0.5B-Instruct` via Transformers on Apple MPS.

### 2.2 Tasks and graders

- **Synthetic math.** Mixed single-step and parenthesised arithmetic with easy/medium/hard labels.
  Answers are graded with numeric tolerance.
- **HumanEval fallback.** Six synthetic function specifications with unit tests executed in a
  restricted subprocess. The real HumanEval loader is implemented and falls back to these problems
  when the dataset is unavailable.
- **ChartQA stand-in.** A tabular chart with categorical/numeric questions, graded with 5% numeric
  tolerance and case-insensitive string matching. The real ChartQA loader is implemented but images
  are unavailable in this CPU-only build.
- **GSM8K.** A real loader and numeric grader are implemented with an offline synthetic fallback; it
  is not part of the committed sweep.

We use 16 problems per seed, three seeds (0, 1, 2), and 48 problem-seed evaluations per strategy cell.
Difficulty is assigned from the pass rate of four pilot samples, cached as JSON: `>=0.75` easy,
`>=0.25` medium, otherwise hard.

### 2.3 Compute accounting

Let `N` be parameters, `P` prompt tokens and `C` completion tokens. We use

```
FLOPs = 2 * N * (P + C)
```

for the forward pass, excluding quadratic attention, verifier scoring and training. Dollar cost uses
$0.15/M prompt tokens and $0.60/M completion tokens. Cost per correct answer is the total token or
dollar cost of a cell divided by its number of correct answers. Confidence intervals are percentile
bootstraps with 2,000 resamples over problem-seed evaluations.

### 2.4 Strategies

- **Greedy:** one deterministic sample.
- **Self-consistency:** N samples at temperature 0.8, plurality vote, ties broken by first occurrence.
- **Best-of-N:** N samples, rank by a verifier, return the argmax.
- **Weighted vote:** softmax of verifier scores summed per extracted answer.
- **Step search:** beam/lookahead over partial reasoning prefixes, with verifier-scored pruning and at
  most `min(3, N)` rounds while never exceeding N expansions.
- **Oracle and length-prior ablations:** the task grader as an upper-bound verifier, and a baseline
  preferring completions near 32 generated words.

### 2.5 Verifier

Training data are 560 samples from 140 math problems (4 samples each), labelled by the grader; the
held-out set is 180 samples from 45 problems. The outcome verifier is a one-layer byte-level
Transformer encoder with a scalar head trained with BCE. The custom pairwise variant uses a logistic
margin loss on correct/incorrect pairs. The TRL variant trains a tiny BERT-style sequence classifier
with `trl.RewardTrainer`. All variants are evaluated with AUROC, threshold accuracy, expected
calibration error (ECE, 10 bins), and top-1 selection accuracy versus random selection.

## 3. Results

### 3.1 Headline accuracy

![Accuracy versus FLOPs](../results/figures/accuracy_vs_flops.png)

**Table 1. Highest-accuracy main-strategy cell per task (synthetic backend).**

| task | model | strategy | N | accuracy (95% CI) | tokens/correct | $/correct |
|---|---|---|---|---|---|---|
| synthetic | mock-qwen2.5-1.5b | weighted_vote | 8 | 1.000 [1.000, 1.000] | 695.5 | 0.0002688 |
| humaneval | mock-qwen2.5-1.5b | self_consistency | 4 | 0.833 [0.729, 0.938] | 297.1 | 0.0000968 |
| chartqa | mock-qwen2.5-1.5b | self_consistency | 8 | 0.938 [0.854, 1.000] | 565.3 | 0.0001270 |

The full per-model strategy-by-budget grid is in `results/tables/headline_grid.md`. On synthetic math,
weighted voting and best-of-N overtake self-consistency; on code, self-consistency at N=4 is strongest;
on chart QA, self-consistency at N=8 is strongest. Best-of-N with the math-trained verifier degrades
with larger N on chart QA, from 0.562 at N=1 to 0.229 at N=16 for the 1.5B model, because it selects
confidently wrong chart answers.

### 3.2 Compute-optimal budgets

![Accuracy versus budget by difficulty](../results/figures/accuracy_vs_budget_difficulty.png)

We choose the best deployable configuration under each task's operating FLOPs cap (chart `4.51e11`,
code `4.20e11`, synthetic `5.95e11`). Table 2 reports the selected configuration per difficulty bucket.

**Table 2. Compute-optimal configuration per task and difficulty bucket.**

| task | difficulty | model | strategy | N | accuracy | mean FLOPs | n |
|---|---|---|---|---|---|---|---|
| chartqa | easy | mock-qwen2.5-0.5b | best_of_n | 2 | 1.000 | 1.308e11 | 10 |
| chartqa | hard | mock-qwen2.5-0.5b | step_search | 3 | 0.600 | 2.476e11 | 5 |
| chartqa | medium | mock-qwen2.5-0.5b | weighted_vote | 4 | 0.576 | 2.619e11 | 33 |
| humaneval | easy | mock-qwen2.5-0.5b | best_of_n | 1 | 1.000 | 6.357e10 | 9 |
| humaneval | hard | mock-qwen2.5-0.5b | best_of_n | 1 | 0.000 | 5.928e10 | 5 |
| humaneval | medium | mock-qwen2.5-1.5b | best_of_n | 1 | 0.909 | 1.971e11 | 22 |
| synthetic | easy | mock-qwen2.5-0.5b | best_of_n | 2 | 1.000 | 1.664e11 | 16 |
| synthetic | hard | mock-qwen2.5-0.5b | step_search | 2 | 0.308 | 2.422e11 | 13 |
| synthetic | medium | mock-qwen2.5-0.5b | best_of_n | 3 | 1.000 | 2.572e11 | 19 |

The selected budgets are small (1–4 samples); larger budgets increase FLOPs without improving the
selected operating point. Hard code problems are a failure mode: the bucket has zero accuracy at the
cap, so more sampling does not solve it.

### 3.3 Parameters versus test-time compute at equal FLOPs

We compare the best 0.5B configuration within +/-25% of the 1.5B greedy compute against the 1.5B greedy
baseline. N=3 is included in the sweep specifically to match the 3.1× parameter ratio.

**Table 3. Equal-FLOPs comparison.**

| task | small configuration | small acc. | big greedy acc. | FLOP ratio | Δ accuracy (95% CI) | verdict |
|---|---|---|---|---|---|---|
| chartqa | self_consistency N=3 | 0.542 | 0.562 | 0.96 | -0.021 [-0.208, +0.167] | statistical tie |
| humaneval | best_of_n N=3 | 0.583 | 0.667 | 0.95 | -0.083 [-0.271, +0.125] | statistical tie |
| synthetic | best_of_n N=3 | 0.729 | 0.604 | 0.96 | +0.125 [-0.042, +0.292] | statistical tie |

![Equal-FLOPs contrast](../results/figures/equal_flops.png)

The point estimates are mixed and every interval crosses zero. At this sample size, the experiment does
not establish a universal advantage for either scaling axis. The synthetic math direction is
encouraging for small-model sampling; the code direction favours the larger model.

### 3.4 Verifier accuracy and headroom

**Table 4. Held-out verifier metrics.**

| objective | AUROC | accuracy | ECE | top-1 | random |
|---|---|---|---|---|---|
| outcome BCE | 0.9952 | 0.9833 | 0.0160 | 0.8667 | 0.5111 |
| pairwise custom | 0.9969 | 0.5111 | 0.3604 | 0.8667 | 0.5111 |
| pairwise TRL | 1.0000 | 0.8722 | 0.3671 | 0.8889 | 0.5111 |

The pairwise variants rank well but are poorly calibrated at a zero threshold. The outcome verifier is
both accurate and well calibrated in-distribution. Figure 4 and Table 5 show the transfer failure:
on code and chart QA the learned verifier is worse than a length prior, and the oracle gap is 0.354 and
0.729 respectively.

![Verifier headroom](../results/figures/verifier_headroom.png)

**Table 5. Best-of-N at N=8.**

| task | learned | length prior | oracle | oracle − learned |
|---|---|---|---|---|
| chartqa | 0.271 | 0.500 | 1.000 | +0.729 |
| humaneval | 0.646 | 1.000 | 1.000 | +0.354 |
| synthetic | 0.906 | 0.927 | 0.927 | +0.021 |

### 3.5 Cost and real serving

![Pareto frontier](../results/figures/pareto_frontier.png)

The committed best cells cost $0.0000968 (code), $0.0001270 (chart QA) and $0.0002688 (synthetic math)
per correct answer under the fixed price constants. A real HF serving check of Qwen2.5-0.5B on Apple
MPS, using self-consistency N=4 over eight problems (32 samples), produced p50 latency 1.152 s, p95
1.871 s, 96.92 completion tokens/s and 0.757 problems/s, with 271.2 MB peak RSS and a 2.74 s load time.
The 32-token cap was hit by all samples, so this is a serving measurement, not an accuracy comparison;
accuracy on the eight synthetic prompts was 0.25.

### 3.6 Multimodal transfer

**Table 6. Transfer of the text-winning strategy to the chart stand-in.**

| metric | value |
|---|---|
| winning synthetic strategy | weighted_vote, N=8 |
| source accuracy | 0.927 [0.750, 1.000] |
| same-strategy chart accuracy | 0.667 [0.438, 0.875] |
| best chart strategy | self_consistency, N=8, 0.802 [0.521, 1.000] |
| greedy chart accuracy | 0.500 [0.292, 0.708] |
| transfer vs greedy | +0.167 |
| transfer vs chart best | -0.135 |
| retention | 0.719 |

Weighted voting improves over greedy on charts but loses to self-consistency. The winning text strategy
does not transfer unchanged.

## 4. Ablations

- **Oracle selection** reaches 1.000 on chart QA and code at N=8, showing that candidate generation is
  not the primary bottleneck; the verifier is.
- **A length prior** reaches 1.000 on code and 0.500 on charts, outperforming the learned outcome
  verifier. This baseline should be reported alongside any learned verifier.
- **Step search** is unstable: it is selected for hard chart and synthetic buckets but often trails
  self-consistency at matched budget because partial-prefix scoring is noisy.
- **Larger budgets are not free:** best-of-N on chart QA decreases from 0.562 (N=1) to 0.229 (N=16) for
  the 1.5B model, a clear negative scaling result caused by verifier error.

## 5. Limitations

The central limitation is that the full sweep uses a simulator, not Qwen weights. The synthetic backend
is useful for testing the measurement apparatus and for studying strategy behaviour with known
correctness rates, but its AUROC and ranking results are not predictive of real model performance. The
incorrect mock completions include a self-doubt phrase, making the verifier’s in-distribution AUROC
optimistic. Code and chart tasks are stand-ins; ChartQA images and real HumanEval solutions were not
used. Each cell has 48 evaluations, so confidence intervals are wide. FLOPs are the standard `2N` per
token approximation, and prices are fixed constants. The serving check uses eight prompts, a 32-token
cap, Apple MPS instead of CUDA, and no vLLM. The sandbox is a research grader, not a security boundary.

Future work should repeat the grid with real Qwen2.5 checkpoints on a GPU, train verifiers on real
outcome data without simulator cues, and evaluate actual GSM8K, HumanEval and ChartQA slices. A
multimodal VLM backend and a stronger process verifier are the natural next steps.

## 6. Reproduction

```bash
pip install -e ".[dev]"
python scripts/write_environment.py
bash scripts/train_verifier.sh
python -m scaletimellm.cli sweep --config configs/full.yaml --no-resume
HF_HOME=.hf_cache python -m scaletimellm.cli bench-serve --config configs/serving.yaml
python -m scaletimellm.cli report
make lint && make test
```

Seeds, model sizes, task slices, prices and budgets are in `configs/full.yaml` and
`configs/serving.yaml`. Raw records are in `results/raw/sweep/`; the analysis is in
`results/aggregate.csv`, `results/compute_optimal.json`, `results/summary.json` and
`results/REPORT.md`.

## References

1. Wei et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
2. Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.
3. Cobbe et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021.
4. Chen et al. Evaluating Large Language Models Trained on Code. arXiv:2107.03374, 2021.
5. Masry et al. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical
   Reasoning. Findings of ACL 2022.
6. Lightman et al. Let’s Verify Step by Step. ICLR 2024.
7. Qwen Team. Qwen2.5 Technical Report. 2024.
8. Hugging Face TRL: Transformer Reinforcement Learning. https://github.com/huggingface/trl
9. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention
   (vLLM). SOSP 2023.
