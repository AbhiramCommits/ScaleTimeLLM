"""Best-of-N selection with a verifier (or an oracle upper bound)."""

from __future__ import annotations

from typing import Any

from scaletimellm.backends.base import LMBackend
from scaletimellm.strategies.base import Strategy, StrategyResult, timed
from scaletimellm.strategies.verifiers import score_many
from scaletimellm.tasks.base import Problem, Task


class BestOfNStrategy(Strategy):
    """Sample N solutions and return the verifier's top-ranked candidate."""

    name = "best_of_n"

    def solve(
        self,
        problem: Problem,
        backend: LMBackend,
        task: Task,
        budget: int,
        seed: int = 0,
    ) -> StrategyResult:
        if self.verifier is None:
            raise ValueError("best_of_n requires a verifier")
        started = timed()
        n = max(1, int(budget))
        prompt = task.prompt(problem)
        generations = backend.generate(
            [prompt],
            n=n,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            seed=seed,
        )[0]
        answers = [task.extract(generation.text) for generation in generations]
        scores = score_many(self.verifier, prompt, [g.text for g in generations])
        best_index = max(range(len(generations)), key=lambda index: (scores[index], -index))
        answer = answers[best_index]
        trace = self._generations_to_trace(generations, answers, scores)
        for entry in trace:
            entry["selected"] = entry["sample"] == best_index
            entry["verifier"] = getattr(self.verifier, "name", type(self.verifier).__name__)
        return self._finalize(
            answer=answer,
            problem=problem,
            task=task,
            generations=generations,
            wall_time_s=timed() - started,
            trace=trace,
            n_params=backend.n_params,
        )


def make_oracle(task: Task, problem: Problem) -> Any:
    """Convenience constructor kept here to avoid import cycles in tests."""
    from scaletimellm.strategies.verifiers import OracleVerifier

    return OracleVerifier(task, problem)
