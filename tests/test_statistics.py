import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
from scipy.stats import friedmanchisquare


def test_welch_anova_gameshowell():
    baseline = np.array([1.0, 1.1, 0.9, 1.2])
    method1 = np.array([2.0, 2.1, 1.9, 2.2])
    method2 = np.array([3.0, 3.1, 2.9, 3.2])
    res = anova_oneway([baseline, method1, method2], use_var="unequal")
    assert res.pvalue < 0.05
    df = pd.DataFrame(
        {
            "score": np.concatenate([baseline, method1, method2]),
            "group": ["A"] * len(baseline)
            + ["B"] * len(method1)
            + ["C"] * len(method2),
        }
    )
    gh = pg.pairwise_gameshowell(dv="score", between="group", data=df)
    pvals = []
    for _, row in gh.iterrows():
        if "A" in (row["A"], row["B"]):
            pvals.append(row["pval"])
    _, padj, _, _ = multipletests(pvals, method="holm")
    assert all(p < 0.05 for p in padj)


def test_friedman_statistic():
    baseline = [1, 2, 3, 4, 5]
    method1 = [2, 3, 4, 5, 6]
    method2 = [3, 4, 5, 6, 7]
    stat, p = friedmanchisquare(baseline, method1, method2)
    assert p < 0.05
