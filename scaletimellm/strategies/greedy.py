"""Greedy single-sample decoding."""

from __future__ import annotations

from scaletimellm.backends.base import LMBackend
from scaletimellm.strategies.base import Strategy, StrategyResult, timed
from scaletimellm.tasks.base import Problem, Task


class GreedyStrategy(Strategy):
    """One deterministic sample; the compute floor for every comparison."""

    name = "greedy"

    def solve(
        self,
        problem: Problem,
        backend: LMBackend,
        task: Task,
        budget: int,
        seed: int = 0,
    ) -> StrategyResult:
        """Generate one deterministic sample."""
        del budget  # greedy always uses one sample
        started = timed()
        generation = backend.generate(
            [task.prompt(problem)],
            n=1,
            temperature=0.0,
            max_new_tokens=self.max_new_tokens,
            seed=seed,
        )[0]
        answer = task.extract(generation[0].text)
        wall_time = timed() - started
        return self._finalize(
            answer=answer,
            problem=problem,
            task=task,
            generations=generation,
            wall_time_s=wall_time,
            trace=self._generations_to_trace(generation, [answer]),
            n_params=backend.n_params,
        )
