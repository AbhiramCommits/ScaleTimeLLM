"""Chart/diagram question answering with a synthetic multimodal stand-in."""

from __future__ import annotations

import random
import re
from typing import Any

from scaletimellm.tasks.answers import extract_last_line, relaxed_string_match
from scaletimellm.tasks.base import Problem, Task
from scaletimellm.tasks.gsm8k import downloads_allowed

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
_PRODUCTS = ["Apples", "Bananas", "Cherries", "Dates", "Elderberry"]


class ChartQATask(Task):
    """ChartQA relaxed-accuracy grading.

    A real ChartQA item is an image plus a question.  Text-only and mock backends
    cannot consume pixels, so when the dataset is unavailable (or when a caller
    requests the offline stand-in) the task renders a chart as an ASCII table.
    This still tests the transfer of a text reasoning strategy to a multimodal-
    style QA interface, but it is explicitly a synthetic stand-in.
    """

    name = "chartqa"
    modality = "multimodal"

    def __init__(self, force_synthetic: bool = False) -> None:
        self.force_synthetic = force_synthetic

    def load(self, n_problems: int, seed: int) -> list[Problem]:
        if downloads_allowed() and not self.force_synthetic:
            try:
                from datasets import load_dataset

                dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
                order = list(range(len(dataset)))
                random.Random(seed).shuffle(order)
                problems: list[Problem] = []
                for index in order[:n_problems]:
                    row: dict[str, Any] = dataset[int(index)]
                    labels = row.get("label", row.get("answer", ""))
                    if isinstance(labels, list):
                        labels = labels[0] if labels else ""
                    question = str(row.get("query", row.get("question", "")))
                    image = row.get("image")
                    prompt = (
                        "Answer the question about the chart. Give a short answer after "
                        f'"Answer:".\n\nQuestion: {question}\n\nAnswer:'
                    )
                    problems.append(
                        Problem(
                            id=f"chartqa-{int(index):04d}",
                            text=question,
                            gold=str(labels),
                            metadata={
                                "source": "chartqa",
                                "task": self.name,
                                "modality": "image",
                                "image": image,
                            },
                            prompt_override=prompt,
                        )
                    )
                if problems:
                    return problems
            except Exception:
                pass
        return self._synthetic(n_problems, seed)

    def _synthetic(self, n_problems: int, seed: int) -> list[Problem]:
        rng = random.Random(seed * 101 + 3)
        problems: list[Problem] = []
        for index in range(n_problems):
            categories = rng.sample(_MONTHS, 4) if index % 2 == 0 else rng.sample(_PRODUCTS, 4)
            values = rng.sample(range(8, 60), len(categories))
            data = dict(zip(categories, values, strict=True))
            kind = index % 4
            if kind == 0:
                question = "Which category has the highest value?"
                answer = max(data, key=data.get)
            elif kind == 1:
                category = categories[rng.randrange(len(categories))]
                question = f"What is the value for {category}?"
                answer = str(data[category])
            elif kind == 2:
                question = "What is the sum of all values?"
                answer = str(sum(values))
            else:
                question = "Which category has the lowest value?"
                answer = min(data, key=data.get)
            table = "\n".join(f"{category}: {value}" for category, value in data.items())
            prompt = (
                "You are given a chart in tabular form (a text stand-in for a chart image).\n"
                f"Chart data:\n{table}\n\nQuestion: {question}\n"
                'Answer with a short phrase after "Answer:".'
            )
            problems.append(
                Problem(
                    id=f"chartqa-synthetic-{seed}-{index:03d}",
                    text=question,
                    gold=answer,
                    metadata={
                        "source": "chartqa_synthetic_standin",
                        "task": self.name,
                        "modality": "synthetic_chart",
                        "chart_data": data,
                        "chart_type": "bar",
                    },
                    prompt_override=prompt,
                )
            )
        return problems

    def prompt(self, problem: Problem) -> str:
        if problem.prompt_override:
            return problem.prompt_override
        return (
            "Answer the question about the chart with a short phrase.\n\n"
            f"Question: {problem.text}\nAnswer:"
        )

    def extract(self, completion: str) -> str | None:
        if not completion:
            return None
        match = re.search(r"Answer:\s*([^\n]+)", completion, flags=re.I)
        if match:
            value = match.group(1).strip().strip("`\"' .")
            return value or None
        return extract_last_line(completion)

    def grade(self, problem: Problem, answer: str | None) -> bool:
        return relaxed_string_match(problem.gold, answer, numeric_tol=0.05)
