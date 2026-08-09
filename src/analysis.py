"""Statistical helpers for the preregistered confirmatory experiment."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp


def mean_confidence_interval(
    values: Iterable[float],
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:
    """Return mean, sample standard deviation and a two-sided t interval."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return (float("nan"),) * 4
    mean = float(array.mean())
    if array.size == 1:
        return mean, 0.0, float("nan"), float("nan")
    std = float(array.std(ddof=1))
    half_width = float(
        t.ppf((1 + confidence) / 2, array.size - 1) * std / np.sqrt(array.size)
    )
    return mean, std, mean - half_width, mean + half_width


def paired_difference(
    baseline: Iterable[float],
    candidate: Iterable[float],
) -> dict[str, float | int]:
    """Summarize paired candidate-minus-baseline differences."""
    baseline_values = np.asarray(list(baseline), dtype=float)
    candidate_values = np.asarray(list(candidate), dtype=float)
    if baseline_values.shape != candidate_values.shape:
        raise ValueError("Paired samples must have the same shape")
    if baseline_values.size < 2:
        return {
            "N": int(baseline_values.size),
            "Mean Difference": float("nan"),
            "Difference CI Low": float("nan"),
            "Difference CI High": float("nan"),
            "Raw p-value": float("nan"),
        }

    differences = candidate_values - baseline_values
    mean, std, ci_low, ci_high = mean_confidence_interval(differences)
    if np.isclose(std, 0.0):
        p_value = 1.0 if np.isclose(mean, 0.0) else 0.0
    else:
        p_value = float(ttest_1samp(differences, popmean=0.0).pvalue)
    return {
        "N": int(differences.size),
        "Mean Difference": mean,
        "Difference CI Low": ci_low,
        "Difference CI High": ci_high,
        "Raw p-value": p_value,
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Apply Holm's step-down family-wise error correction."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted.tolist()

    order = finite_indices[np.argsort(values[finite_indices])]
    running_max = 0.0
    family_size = len(order)
    for rank, index in enumerate(order):
        corrected = min(1.0, (family_size - rank) * values[index])
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
    return adjusted.tolist()


def confirmatory_comparisons(
    runs: pd.DataFrame,
    metrics: list[str],
    hybrid: str = "icm_planner",
    comparators: list[str] | None = None,
    minimum_effect: float = 0.0,
) -> pd.DataFrame:
    """Compare the hybrid against every preregistered comparator by seed."""
    if comparators is None:
        comparators = [
            variant for variant in runs["Variant"].unique() if variant != hybrid
        ]

    rows = []
    hybrid_rows = runs[runs["Variant"] == hybrid]
    for metric in metrics:
        metric_rows = []
        for comparator in comparators:
            comparator_rows = runs[runs["Variant"] == comparator]
            paired = comparator_rows[["Seed", metric]].merge(
                hybrid_rows[["Seed", metric]],
                on="Seed",
                suffixes=("_comparator", "_hybrid"),
            )
            result = paired_difference(
                paired[f"{metric}_comparator"],
                paired[f"{metric}_hybrid"],
            )
            result.update(
                {
                    "Metric": metric,
                    "Hybrid": hybrid,
                    "Comparator": comparator,
                    "Hybrid Mean": float(paired[f"{metric}_hybrid"].mean()),
                    "Comparator Mean": float(paired[f"{metric}_comparator"].mean()),
                    "Minimum Meaningful Effect": minimum_effect,
                }
            )
            metric_rows.append(result)

        adjusted = holm_adjust(row["Raw p-value"] for row in metric_rows)
        for row, adjusted_p in zip(metric_rows, adjusted, strict=True):
            row["Holm-adjusted p-value"] = adjusted_p
            row["Statistically Superior"] = bool(
                np.isfinite(adjusted_p)
                and adjusted_p < 0.05
                and row["Difference CI Low"] > 0
            )
            row["Practically Meaningful"] = bool(
                row["Mean Difference"] >= minimum_effect
            )
            row["Superiority Criterion Met"] = bool(
                row["Statistically Superior"] and row["Practically Meaningful"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def factorial_interaction(
    runs: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Test the paired ICM-by-planner interaction for the core 2x2 ablation."""
    required = ["ppo", "icm", "planner", "icm_planner"]
    pivot = runs[runs["Variant"].isin(required)].pivot(
        index="Seed",
        columns="Variant",
        values=metric,
    )
    pivot = pivot.reindex(columns=required)
    pivot = pivot.dropna(subset=required)
    interaction = pivot["icm_planner"] - pivot["icm"] - pivot["planner"] + pivot["ppo"]
    mean, std, ci_low, ci_high = mean_confidence_interval(interaction)
    if len(interaction) < 2:
        p_value = float("nan")
    elif np.isclose(std, 0.0):
        p_value = 1.0 if np.isclose(mean, 0.0) else 0.0
    else:
        p_value = float(ttest_1samp(interaction, popmean=0.0).pvalue)
    return pd.DataFrame(
        [
            {
                "Metric": metric,
                "N": len(interaction),
                "ICM x Planner Interaction": mean,
                "Interaction CI Low": ci_low,
                "Interaction CI High": ci_high,
                "p-value": p_value,
                "Positive Synergy": bool(
                    np.isfinite(p_value) and p_value < 0.05 and ci_low > 0
                ),
            }
        ]
    )
