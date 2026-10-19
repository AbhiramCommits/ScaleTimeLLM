"""Verifier-weighted majority voting."""

from __future__ import annotations

from collections import OrderedDict

from scaletimellm.backends.base import LMBackend
from scaletimellm.strategies.base import Strategy, StrategyResult, timed
from scaletimellm.strategies.verifiers import score_many, softmax_weights
from scaletimellm.tasks.base import Problem, Task


class WeightedVoteStrategy(Strategy):
    """Combine self-consistency with verifier confidence.

    Each sample receives a stable softmax weight derived from its verifier score;
    weights are summed per extracted answer.  Ties go to the answer seen first.
    With an oracle verifier this reduces to selecting a correct answer; with the
    length prior it tests whether confidence adds anything over count voting.
    """

    name = "weighted_vote"

    def solve(
        self,
        problem: Problem,
        backend: LMBackend,
        task: Task,
        budget: int,
        seed: int = 0,
    ) -> StrategyResult:
        if self.verifier is None:
            raise ValueError("weighted_vote requires a verifier")
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
        weights = softmax_weights(scores)
        answer_weights: dict[str, float] = {}
        first_seen: OrderedDict[str, int] = OrderedDict()
        for index, (answer, weight) in enumerate(zip(answers, weights, strict=True)):
            if answer is None:
                continue
            answer_weights[answer] = answer_weights.get(answer, 0.0) + weight
            first_seen.setdefault(answer, index)
        answer = None
        if answer_weights:
            maximum = max(answer_weights.values())
            tied = [candidate for candidate, value in answer_weights.items() if value == maximum]
            answer = min(tied, key=lambda candidate: first_seen[candidate])
        trace = self._generations_to_trace(generations, answers, scores)
        for entry in trace:
            entry["vote_weight"] = weights[entry["sample"]]
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
