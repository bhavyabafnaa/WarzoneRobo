import pandas as pd

from src.analysis import (
    confirmatory_comparisons,
    factorial_interaction,
    holm_adjust,
)


def test_holm_adjust_controls_the_family():
    assert holm_adjust([0.01, 0.03, 0.04]) == [0.03, 0.06, 0.06]


def test_confirmatory_superiority_requires_every_criterion():
    rows = []
    values = {
        "ppo": [0.0, 0.1, -0.1],
        "icm": [0.2, 0.3, 0.1],
        "icm_planner": [1.2, 1.3, 1.1],
    }
    for variant, metrics in values.items():
        for seed, metric in enumerate(metrics):
            rows.append({"Seed": seed, "Variant": variant, "Test Reward Mean": metric})

    result = confirmatory_comparisons(
        pd.DataFrame(rows),
        metrics=["Test Reward Mean"],
        hybrid="icm_planner",
        comparators=["ppo", "icm"],
        minimum_effect=0.5,
    )

    assert len(result) == 2
    assert result["Superiority Criterion Met"].all()


def test_factorial_interaction_detects_positive_synergy():
    rows = []
    values = {"ppo": 0.0, "icm": 1.0, "planner": 1.0, "icm_planner": 3.0}
    for seed in range(3):
        for variant, value in values.items():
            rows.append(
                {
                    "Seed": seed,
                    "Variant": variant,
                    "Test Reward Mean": value + seed * 0.1,
                }
            )

    result = factorial_interaction(pd.DataFrame(rows), "Test Reward Mean").iloc[0]

    assert result["ICM x Planner Interaction"] == 1.0
    assert bool(result["Positive Synergy"]) is True
