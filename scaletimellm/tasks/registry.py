"""Task registry."""

from __future__ import annotations

from typing import Any

from scaletimellm.tasks.base import Problem, Task
from scaletimellm.tasks.chartqa import ChartQATask
from scaletimellm.tasks.gsm8k import GSM8KTask
from scaletimellm.tasks.humaneval import HumanEvalTask
from scaletimellm.tasks.synthetic import SyntheticTask


def get_task(name: str, **kwargs: Any) -> Task:
    """Instantiate a task by name."""
    key = name.lower().strip()
    if key in {"synthetic", "math", "synthetic_math"}:
        return SyntheticTask()
    if key in {"gsm8k", "gsm"}:
        return GSM8KTask()
    if key in {"humaneval", "human_eval", "code"}:
        return HumanEvalTask(**kwargs)
    if key in {"chartqa", "chart", "multimodal"}:
        return ChartQATask(**kwargs)
    raise ValueError(f"unknown task: {name!r}")


__all__ = ["Problem", "Task", "get_task"]
