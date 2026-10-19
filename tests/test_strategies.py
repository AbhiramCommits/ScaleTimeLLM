"""Strategy tests using the deterministic mock backend."""

from __future__ import annotations

from scaletimellm.backends.mock import MockLM
from scaletimellm.strategies.best_of_n import BestOfNStrategy
from scaletimellm.strategies.greedy import GreedyStrategy
from scaletimellm.strategies.self_consistency import SelfConsistencyStrategy, majority_vote
from scaletimellm.strategies.step_search import StepSearchStrategy
from scaletimellm.strategies.verifiers import ConstantVerifier, OracleVerifier
from scaletimellm.strategies.weighted_vote import WeightedVoteStrategy
from scaletimellm.tasks.synthetic import SyntheticTask


def _setup(n_problems: int = 120, accuracy: float = 0.4):
    task = SyntheticTask()
    problems = task.load(n_problems, seed=0)
    backend = MockLM(
        n_params=32_000_000,
        accuracy=accuracy,
        seed=0,
        difficulty_adjustment=False,
    )
    return task, problems, backend


def test_majority_vote_tie_break_first_occurrence() -> None:
    assert majority_vote(["a", "b", "b", "a"]) == "a"
    assert majority_vote([None, "x", None, "x"]) == "x"
    assert majority_vote([None, None]) is None


def test_self_consistency_beats_greedy_at_low_accuracy() -> None:
    task, problems, backend = _setup()
    greedy = GreedyStrategy(max_new_tokens=64)
    consistency = SelfConsistencyStrategy(max_new_tokens=64, temperature=0.8)
    greedy_correct = 0
    consistency_correct = 0
    for problem in problems:
        greedy_correct += greedy.solve(problem, backend, task, 1, seed=0).is_correct
        consistency_correct += consistency.solve(problem, backend, task, 9, seed=0).is_correct
    greedy_accuracy = greedy_correct / len(problems)
    consistency_accuracy = consistency_correct / len(problems)
    assert 0.30 <= greedy_accuracy <= 0.50
    assert consistency_accuracy > greedy_accuracy


def test_flops_grow_linearly_with_budget() -> None:
    task, problems, backend = _setup(n_problems=20)
    strategy = SelfConsistencyStrategy(max_new_tokens=32, temperature=0.8)
    flops = {}
    for budget in (1, 2, 4, 8):
        total = sum(
            strategy.solve(problem, backend, task, budget, seed=0).flops for problem in problems
        )
        flops[budget] = total
    for budget in (2, 4, 8):
        ratio = flops[budget] / flops[1]
        assert abs(ratio - budget) / budget < 0.05


def test_strategies_are_deterministic_for_fixed_seed() -> None:
    task, problems, backend = _setup(n_problems=8)
    problem = problems[0]
    for strategy in (
        GreedyStrategy(max_new_tokens=32),
        SelfConsistencyStrategy(max_new_tokens=32, temperature=0.8),
        BestOfNStrategy(max_new_tokens=32, temperature=0.8),
        WeightedVoteStrategy(max_new_tokens=32, temperature=0.8),
        StepSearchStrategy(max_new_tokens=32, temperature=0.8, max_steps=3),
    ):
        verifier = (
            OracleVerifier(task, problem)
            if strategy.name in {"best_of_n", "weighted_vote", "step_search"}
            else None
        )
        strategy.verifier = verifier
        first = strategy.solve(problem, backend, task, 4, seed=0)
        second = strategy.solve(problem, backend, task, 4, seed=0)
        assert first.answer == second.answer
        assert first.prompt_tokens == second.prompt_tokens
        assert first.completion_tokens == second.completion_tokens
        assert first.flops == second.flops


def test_oracle_best_of_n_dominates_constant_verifier() -> None:
    task, problems, backend = _setup(n_problems=20)
    oracle = BestOfNStrategy(max_new_tokens=32, temperature=0.8)
    constant = BestOfNStrategy(
        max_new_tokens=32, temperature=0.8, verifier=ConstantVerifier()
    )
    oracle_correct = 0
    constant_correct = 0
    for problem in problems:
        oracle.verifier = OracleVerifier(task, problem)
        oracle_correct += oracle.solve(problem, backend, task, 8, seed=0).is_correct
        constant_correct += constant.solve(problem, backend, task, 8, seed=0).is_correct
    assert oracle_correct > constant_correct


def test_step_search_respects_budget_and_tokens() -> None:
    task, problems, backend = _setup(n_problems=4)
    problem = problems[0]
    strategy = StepSearchStrategy(max_new_tokens=32, temperature=0.8, verifier=None)
    strategy.verifier = OracleVerifier(task, problem)
    result = strategy.solve(problem, backend, task, 8, seed=0)
    assert 1 <= result.n_samples <= 8
    assert result.prompt_tokens > 0 and result.completion_tokens > 0
    assert result.flops > 0
    assert result.wall_time_s >= 0
