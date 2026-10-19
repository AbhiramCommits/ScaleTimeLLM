"""Self-consistency: sample N and take a plurality vote."""

from __future__ import annotations

from collections import Counter, OrderedDict

from scaletimellm.backends.base import LMBackend
from scaletimellm.strategies.base import Strategy, StrategyResult, timed
from scaletimellm.tasks.base import Problem, Task


def majority_vote(answers: list[str | None]) -> str | None:
    """Return the most frequent non-null answer.

    Ties are broken by first occurrence in the sample list, which is fully
    deterministic for a fixed seed and sample order.
    """
    valid = [answer for answer in answers if answer is not None]
    if not valid:
        return None
    counts = Counter(valid)
    first_seen: OrderedDict[str, int] = OrderedDict()
    for index, answer in enumerate(valid):
        first_seen.setdefault(answer, index)
    best_count = max(counts.values())
    tied = [answer for answer, count in counts.items() if count == best_count]
    return min(tied, key=lambda answer: first_seen[answer])


class SelfConsistencyStrategy(Strategy):
    """Sample ``budget`` independent solutions and plurality-vote."""

    name = "self_consistency"

    def solve(
        self,
        problem: Problem,
        backend: LMBackend,
        task: Task,
        budget: int,
        seed: int = 0,
    ) -> StrategyResult:
        """Sample N solutions and take a plurality vote."""
        started = timed()
        n = max(1, int(budget))
        generations = backend.generate(
            [task.prompt(problem)],
            n=n,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            seed=seed,
        )[0]
        answers = [task.extract(generation.text) for generation in generations]
        answer = majority_vote(answers)
        counts = Counter(answer for answer in answers if answer is not None)
        trace = self._generations_to_trace(generations, answers)
        for entry in trace:
            entry["vote_count"] = counts.get(entry["extracted"], 0)
        return self._finalize(
            answer=answer,
            problem=problem,
            task=task,
            generations=generations,
            wall_time_s=timed() - started,
            trace=trace,
            n_params=backend.n_params,
        )
