"""Step-level beam search over partial reasoning traces."""

from __future__ import annotations

from typing import Any

from scaletimellm.backends.base import Generation, LMBackend
from scaletimellm.strategies.base import Strategy, StrategyResult, timed
from scaletimellm.strategies.verifiers import score_many
from scaletimellm.tasks.base import Problem, Task


class StepSearchStrategy(Strategy):
    """Beam search over reasoning prefixes scored by the verifier.

    The sampling budget caps the total number of expansions (it is never
    exceeded).  Each round expands every retained prefix, scores all new
    prefixes, and keeps ``beam_width`` of them.  The final winner is the highest
    scored prefix that yields an extractable answer, falling back to the best
    prefix overall.
    """

    name = "step_search"

    def __init__(self, max_steps: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_steps = max(1, int(max_steps))

    def solve(
        self,
        problem: Problem,
        backend: LMBackend,
        task: Task,
        budget: int,
        seed: int = 0,
    ) -> StrategyResult:
        """Run verifier-guided step-level beam search."""
        if self.verifier is None:
            raise ValueError("step_search requires a verifier")
        started = timed()
        budget = max(1, int(budget))
        steps = min(self.max_steps, budget)
        beam_width = max(1, budget // (2 * steps))
        expansions_per_beam = max(1, budget // (steps * beam_width))
        base_prompt = task.prompt(problem)
        beams: list[tuple[str, float]] = [("", float("-inf"))]
        all_generations: list[Generation] = []
        all_prefixes: list[str] = []
        all_scores: list[float] = []
        for step in range(steps):
            candidates: list[tuple[str, float]] = []
            for prefix, _ in beams:
                if step == 0:
                    prompt = base_prompt
                else:
                    prompt = (
                        base_prompt
                        + "\n\nPartial solution so far:\n"
                        + prefix
                        + "\nContinue the solution and finish with the final answer."
                    )
                generations = backend.generate(
                    [prompt],
                    n=expansions_per_beam,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    seed=seed + step,
                )[0]
                texts = [generation.text for generation in generations]
                scores = score_many(self.verifier, base_prompt, texts)
                for generation, text, score in zip(generations, texts, scores, strict=True):
                    all_generations.append(generation)
                    all_prefixes.append(text)
                    all_scores.append(score)
                    candidates.append((text, score))
            candidates.sort(key=lambda item: item[1], reverse=True)
            beams = candidates[:beam_width]
        best_prefix, best_score = max(beams, key=lambda item: item[1])
        answer = task.extract(best_prefix)
        if answer is None:
            for prefix, _ in sorted(beams, key=lambda item: item[1], reverse=True):
                answer = task.extract(prefix)
                if answer is not None:
                    best_prefix, best_score = prefix, dict(beams).get(prefix, best_score)
                    break
        trace = self._generations_to_trace(all_generations, all_prefixes, all_scores)
        for entry in trace:
            entry["selected"] = entry["text"] == best_prefix
            entry["verifier"] = getattr(self.verifier, "name", type(self.verifier).__name__)
        return self._finalize(
            answer=answer,
            problem=problem,
            task=task,
            generations=all_generations,
            wall_time_s=timed() - started,
            trace=trace,
            n_params=backend.n_params,
        )
