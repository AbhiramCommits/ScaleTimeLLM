"""Held-out verifier metrics: AUROC, accuracy, ECE and top-1 selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _auroc(labels: list[int], scores: list[float]) -> float:
    """Rank-based AUROC with 0.5 for a degenerate label set."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            tied = np.where(inverse == index)[0]
            ranks[tied] = ranks[tied].mean()
    positive_rank_sum = ranks[np.asarray(labels, dtype=bool)].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def _ece(labels: list[int], scores: list[float], bins: int = 10) -> float:
    if not labels:
        return 0.0
    probabilities = np.asarray([1.0 / (1.0 + np.exp(-score)) for score in scores])
    labels_array = np.asarray(labels)
    total = len(labels)
    value = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (probabilities >= lower) & (
            (probabilities < upper) if upper < 1.0 else (probabilities <= upper)
        )
        if not mask.any():
            continue
        confidence = probabilities[mask].mean()
        accuracy = labels_array[mask].mean()
        value += (mask.sum() / total) * abs(confidence - accuracy)
    return float(value)


def top1_selection(
    rows: list[dict[str, Any]], scores: list[float]
) -> tuple[float, float, int]:
    """Return top-1 selection accuracy, random-selection expectation and n."""
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[str(row["problem_id"])].append((float(score), int(row["label"])))
    if not grouped:
        return 0.0, 0.0, 0
    correct = 0
    random_expected = 0.0
    for candidates in grouped.values():
        best = max(candidates, key=lambda item: item[0])
        correct += best[1]
        random_expected += sum(label for _, label in candidates) / len(candidates)
    return correct / len(grouped), random_expected / len(grouped), len(grouped)


def evaluate_scores(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    """Compute metrics from rows and aligned verifier scores."""
    labels = [int(row["label"]) for row in rows]
    probabilities = [1.0 / (1.0 + np.exp(-score)) for score in scores]
    predictions = [int(probability >= 0.5) for probability in probabilities]
    top1, random_expected, n_problems = top1_selection(rows, scores)
    return {
        "n": len(rows),
        "n_problems": n_problems,
        "positive_rate": sum(labels) / max(1, len(labels)),
        "auroc": _auroc(labels, scores),
        "accuracy": sum(
            prediction == label for prediction, label in zip(predictions, labels, strict=True)
        )
        / max(1, len(labels)),
        "ece": _ece(labels, scores),
        "top1_selection_accuracy": top1,
        "random_selection_accuracy": random_expected,
        "top1_minus_random": top1 - random_expected,
        "mean_score_positive": float(
            np.mean([score for score, label in zip(scores, labels, strict=True) if label == 1])
        )
        if any(labels)
        else 0.0,
        "mean_score_negative": float(
            np.mean([score for score, label in zip(scores, labels, strict=True) if label == 0])
        )
        if any(label == 0 for label in labels)
        else 0.0,
    }


def evaluate_verifier(verifier: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score held-out rows with a verifier and compute metrics."""
    scores: list[float] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["problem_id"])].append(row)
    for problem_rows in grouped.values():
        prompt = str(problem_rows[0]["prompt"])
        completions = [str(row["completion"]) for row in problem_rows]
        scores.extend(verifier.score_many(prompt, completions))
    # The previous loop groups rows, so realign scores to original row order.
    by_id = {str(row["problem_id"]): [] for row in rows}
    cursor = 0
    for problem_id in by_id:
        count = len(grouped[problem_id])
        by_id[problem_id] = scores[cursor : cursor + count]
        cursor += count
    aligned: list[float] = []
    seen: dict[str, int] = defaultdict(int)
    for row in rows:
        problem_id = str(row["problem_id"])
        index = seen[problem_id]
        aligned.append(by_id[problem_id][index])
        seen[problem_id] += 1
    return evaluate_scores(rows, aligned)


def metrics_from_labels(labels: list[int], scores: list[float]) -> dict[str, Any]:
    """Public helper for analytic unit tests."""
    rows = [
        {"label": label, "problem_id": f"p{index}"}
        for index, label in enumerate(labels)
    ]
    return evaluate_scores(rows, scores)
