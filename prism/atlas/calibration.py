"""Uncertainty calibration metrics for PRISM-Atlas response surfaces."""

from __future__ import annotations

import math
from statistics import mean


def calibration_report(
    *,
    predicted_mean: list[float],
    predicted_uncertainty: list[float],
    measured_damage: list[float],
    bins: int = 10,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    if not (len(predicted_mean) == len(predicted_uncertainty) == len(measured_damage)):
        raise ValueError("predicted_mean, predicted_uncertainty, and measured_damage must have equal length.")
    if not predicted_mean:
        raise ValueError("Cannot calibrate uncertainty on an empty set.")

    nll_values = []
    normalized_errors = []
    for mu, sigma, y in zip(predicted_mean, predicted_uncertainty, measured_damage, strict=True):
        sigma = max(float(sigma), 1e-8)
        err = float(y) - float(mu)
        nll_values.append(0.5 * (math.log(2.0 * math.pi * sigma * sigma) + (err * err) / (sigma * sigma)))
        normalized_errors.append(abs(err) / sigma)

    risk_scores = [float(mu) + float(sigma) for mu, sigma in zip(predicted_mean, predicted_uncertainty, strict=True)]
    return {
        "nll": float(mean(nll_values)),
        "spearman": _spearman(list(predicted_mean), list(measured_damage)),
        "risk_ece": _risk_ece(risk_scores, list(measured_damage), bins=max(1, int(bins))),
        "top_k_risky_recall": _top_k_recall(risk_scores, list(measured_damage), top_fraction=top_fraction),
        "mean_abs_normalized_error": float(mean(normalized_errors)),
    }


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = rank
        i = j + 1
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 1.0
    ra = _rank(a)
    rb = _rank(b)
    ma = mean(ra)
    mb = mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(num / (da * db))


def _risk_ece(risk_scores: list[float], measured_damage: list[float], *, bins: int) -> float:
    if not risk_scores:
        return 0.0
    lo = min(risk_scores)
    hi = max(risk_scores)
    if hi <= lo:
        return abs(mean(risk_scores) - mean(measured_damage))
    total = 0.0
    for idx in range(bins):
        left = lo + (hi - lo) * idx / bins
        right = lo + (hi - lo) * (idx + 1) / bins
        members = [
            i
            for i, score in enumerate(risk_scores)
            if (left <= score < right) or (idx == bins - 1 and score <= right)
        ]
        if not members:
            continue
        pred = mean(risk_scores[i] for i in members)
        obs = mean(measured_damage[i] for i in members)
        total += len(members) / len(risk_scores) * abs(pred - obs)
    return float(total)


def _top_k_recall(risk_scores: list[float], measured_damage: list[float], *, top_fraction: float) -> float:
    k = max(1, int(len(risk_scores) * max(0.0, min(1.0, float(top_fraction)))))
    pred_top = {idx for idx, _ in sorted(enumerate(risk_scores), key=lambda item: item[1], reverse=True)[:k]}
    true_top = {idx for idx, _ in sorted(enumerate(measured_damage), key=lambda item: item[1], reverse=True)[:k]}
    return float(len(pred_top & true_top) / max(1, len(true_top)))
