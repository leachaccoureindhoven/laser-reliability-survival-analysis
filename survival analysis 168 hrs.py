# -*- coding: utf-8 -*-
"""
168-h LANDMARK SURVIVAL ANALYSIS -- FULLY CORRECTED VERSION
==============================================================


Laser reliability analysis
168 h landmark Cox proportional-hazards analysis

MAIN RESULTS
------------

Cox Model 4:
    1. Stress current density
    2. Early ITH degradation rate
    3. Cavity length
    4. Temperature
    5. Distance to edge
    6. ITH_0

Representative figures:

    Figure 1:
        Early ITH degradation rate
        Failed vs Censored

    Figure 2:
        Cavity length
        Failed vs Censored

    Figure 3:
        Stress current density
        Failed vs Censored

    Figure 4:
        Early ITH degradation rate
        Failed vs Censored

    Figure 5:
        Martingale residuals by wafer
        Broad vs Strict

COHORT DEFINITIONS
------------------

Broad:
    - >= 3 ITH measurements between 0 and 168 h
    - valid ITH_0
    - observed through 168 h
    - no failure at or before 168 h
    - exact ITH_168 is NOT required

Strict:
    - same requirements as Broad
    - exact ITH_168 required

LANDMARK ANALYSIS
-----------------

Time origin = 168 h.

Failure:
    event = 1
    duration = failure time

Censored:
    event = 0
    duration = last available ITH observation


--------------------------------------------------------------
Rather than swapping the full 6-variable Model 4 for a reduced
model when EPV is low, this script:
    (a) ALWAYS fits and reports the full Model 4, clearly flagged as
        exploratory when EPV < MIN_EPV,
    (b) ALSO fits a reduced current-density-only sensitivity model,
so nothing is hidden -- both results are visible and the reader can judge
for themselves.

@author: Léa Chaccour 
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    theilslopes,
    kruskal,
    spearmanr,
    chi2_contingency,
    mannwhitneyu
)

from scipy.stats import chi2 as chi2dist

from lifelines import CoxPHFitter, KaplanMeierFitter


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    r"C:\Users\20230289\Downloads"
    r"\1Wafer all merged _kept_FEATURE_TABLE_FAIL_ANALYSIS_ITH0"
    r"_with_ProductIDDC_with_edge.csv"
)

WINDOW_H = 168
MIN_EARLY_POINTS = 3
PENALIZER = 0.05
MIN_STD_THRESHOLD = 1e-8

# EPV threshold used only as a WARNING label -- the full Model 4 is still
# fit and reported regardless (see design note above).
MIN_EPV = 5


# ============================================================
# COLUMN NAMES
# ============================================================

COL_FAIL = "FAIL_ITH0"
COL_FAIL_TIME = "FAIL_ITH0_milestone"
COL_ITH0 = "ITH_0"
COL_WAFER = "Wafer number"
COL_CAVITY = "cavity length"
COL_DISTANCE = "distance_to_edge"
COL_TEMP = "Climate_Chamber_Ambient_Temperature [degC]"
COL_CURRENT = "Stress_Current_Density [kA/cm2]"

COX_VARIABLES = [
    COL_CURRENT,
    "early_rate",
    COL_CAVITY,
    COL_TEMP,
    COL_DISTANCE,
    COL_ITH0,
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)
    print(f"Dataset shape: {df.shape}")

    required = [
        COL_FAIL,
        COL_FAIL_TIME,
        COL_ITH0,
        COL_WAFER,
        COL_CAVITY,
        COL_DISTANCE,
        COL_TEMP,
        COL_CURRENT
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # Auto-discover ALL ITH_<time> columns
    ith_times = {}

    for c in df.columns:
        match = re.fullmatch(r"ITH_(\d+)", c.strip())

        if match:
            ith_times[int(match.group(1))] = c

    ith_times = dict(sorted(ith_times.items()))

    print(f"\nAuto-discovered {len(ith_times)} ITH timepoint columns:")
    print(f"  {list(ith_times.keys())}")

    if not ith_times:
        raise ValueError("No ITH_<time> columns found.")

    if len(ith_times) < MIN_EARLY_POINTS:
        raise ValueError("Fewer than 3 ITH timepoints available.")

    max_t = max(ith_times.keys())

    if max_t <= WINDOW_H:
        print(
            f"\nWARNING: no ITH measurements beyond {WINDOW_H}h were found."
        )
        print(
            "Broad and Strict cohorts may therefore be identical."
        )

    return df, ith_times


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df, ith_times):

    df = df.copy()

    def n_early_points(row):

        n = 0

        for t, col in ith_times.items():

            if t <= WINDOW_H:

                value = pd.to_numeric(
                    row[col],
                    errors="coerce"
                )

                if pd.notna(value):
                    n += 1

        return n

    def last_obs(row):

        # Uses ALL ITH timepoints, not just <= WINDOW_H.
        observed_times = []

        for t, col in ith_times.items():

            value = pd.to_numeric(
                row[col],
                errors="coerce"
            )

            if pd.notna(value):
                observed_times.append(t)

        return max(observed_times) if observed_times else np.nan

    def early_rate_and_noise(row):

        x, y = [], []

        for t, col in ith_times.items():

            if t <= WINDOW_H:

                value = pd.to_numeric(
                    row[col],
                    errors="coerce"
                )

                if pd.notna(value):

                    x.append(t)
                    y.append(value)

        if len(x) < MIN_EARLY_POINTS:
            return np.nan, np.nan

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        try:

            slope, intercept, _, _ = theilslopes(y, x)

            predicted = intercept + slope * x
            residuals = y - predicted

            noise_raw = np.std(
                residuals,
                ddof=1
            )

            ith0 = pd.to_numeric(
                row[COL_ITH0],
                errors="coerce"
            )

            noise = (
                np.nan
                if (pd.isna(ith0) or ith0 == 0)
                else noise_raw / abs(ith0)
            )

            return slope, noise

        except Exception:

            return np.nan, np.nan

    df["n_early_points"] = df.apply(
        n_early_points,
        axis=1
    )

    df["last_obs"] = df.apply(
        last_obs,
        axis=1
    )

    results = df.apply(
        lambda row: pd.Series(
            early_rate_and_noise(row)
        ),
        axis=1
    )

    df["early_rate"] = results[0]
    df["extraction_noise"] = results[1]

    return df


# ============================================================
# COHORT DIAGNOSTICS
# ============================================================

def diagnose_cohort_steps(data):

    s = data.copy()

    print("\n" + "=" * 80)
    print("COHORT FILTER DIAGNOSTICS")
    print("=" * 80)

    print(f"Initial dataset: {len(s)}")

    s = s[
        pd.to_numeric(
            s[COL_ITH0],
            errors="coerce"
        ).notna()
    ].copy()

    print(f"After valid ITH_0: {len(s)}")

    s = s[
        s["n_early_points"] >= MIN_EARLY_POINTS
    ].copy()

    print(
        f"After >= {MIN_EARLY_POINTS} early ITH points: {len(s)}"
    )

    fail = pd.to_numeric(
        s[COL_FAIL],
        errors="coerce"
    )

    fail_time = pd.to_numeric(
        s[COL_FAIL_TIME],
        errors="coerce"
    )

    early_failure = (
        fail.eq(1)
        & fail_time.notna()
        & fail_time.le(WINDOW_H)
    )

    print(
        f"Early failures removed (<= {WINDOW_H}h): "
        f"{int(early_failure.sum())}"
    )

    s = s[~early_failure].copy()

    print(
        f"After removing early failures: {len(s)}"
    )

    s168 = s[
        s["last_obs"] >= WINDOW_H
    ].copy()

    print(
        f"\nBROAD {WINDOW_H}h LANDMARK COHORT "
        f"(last_obs >= {WINDOW_H}h, any timepoint)"
    )

    print(f"N = {len(s168)}")

    fail168 = pd.to_numeric(
        s168[COL_FAIL],
        errors="coerce"
    )

    failtime168 = pd.to_numeric(
        s168[COL_FAIL_TIME],
        errors="coerce"
    )

    events168 = (
        fail168.eq(1)
        & failtime168.notna()
        & failtime168.gt(WINDOW_H)
    )

    s168["event"] = events168.astype(int)

    s168["duration"] = np.where(
        s168["event"] == 1,
        failtime168,
        s168["last_obs"]
    )

    print(
        f"Events = {int(events168.sum())}"
    )

    print(
        f"Removed because last_obs < {WINDOW_H}h: "
        f"{len(s) - len(s168)}"
    )

    if "ITH_168" not in s168.columns:
        raise KeyError(
            "ITH_168 is required for strict cohort."
        )

    strict = s168[
        pd.to_numeric(
            s168["ITH_168"],
            errors="coerce"
        ).notna()
    ].copy()

    print(
        f"\nSTRICT COHORT (+ exact ITH_168 required)"
    )

    print(f"N = {len(strict)}")

    fail_s = pd.to_numeric(
        strict[COL_FAIL],
        errors="coerce"
    )

    failtime_s = pd.to_numeric(
        strict[COL_FAIL_TIME],
        errors="coerce"
    )

    events_s = (
        fail_s.eq(1)
        & failtime_s.notna()
        & failtime_s.gt(WINDOW_H)
    )

    strict["event"] = events_s.astype(int)

    strict["duration"] = np.where(
        strict["event"] == 1,
        failtime_s,
        strict["last_obs"]
    )

    print(
        f"Events = {int(events_s.sum())}"
    )

    print(
        "Removed because ITH_168 missing "
        "(but confirmed alive via later reading): "
        f"{len(s168) - len(strict)}"
    )

    return s168, strict


# ============================================================
# BUILD COHORT
# ============================================================

def build_cohort(data, strict=False):

    sub = data.copy()

    sub = sub[
        pd.to_numeric(
            sub[COL_ITH0],
            errors="coerce"
        ).notna()
    ].copy()

    sub = sub[
        sub["n_early_points"] >= MIN_EARLY_POINTS
    ].copy()

    fail = pd.to_numeric(
        sub[COL_FAIL],
        errors="coerce"
    )

    fail_time = pd.to_numeric(
        sub[COL_FAIL_TIME],
        errors="coerce"
    )

    early_failure = (
        fail.eq(1)
        & fail_time.notna()
        & fail_time.le(WINDOW_H)
    )

    sub = sub[~early_failure].copy()

    sub = sub[
        sub["last_obs"] >= WINDOW_H
    ].copy()

    if strict:

        if "ITH_168" not in sub.columns:
            raise KeyError(
                "ITH_168 is required for the strict cohort."
            )

        sub = sub[
            pd.to_numeric(
                sub["ITH_168"],
                errors="coerce"
            ).notna()
        ].copy()

    fail = pd.to_numeric(
        sub[COL_FAIL],
        errors="coerce"
    )

    fail_time = pd.to_numeric(
        sub[COL_FAIL_TIME],
        errors="coerce"
    )

    sub["event"] = (
        fail.eq(1)
        & fail_time.notna()
        & fail_time.gt(WINDOW_H)
    ).astype(int)

    sub["duration"] = np.where(
        sub["event"] == 1,
        fail_time,
        sub["last_obs"]
    )

    sub = sub[
        pd.to_numeric(
            sub["duration"],
            errors="coerce"
        ).notna()
    ].copy()

    sub = sub[
        sub["duration"] >= WINDOW_H
    ].copy()

    return sub


# ============================================================
# INFORMATIVE CENSORING CHECK
# ============================================================

def informative_censoring_check(cohort):

    censored = cohort[
        cohort["event"] == 0
    ].dropna(
        subset=["early_rate", "last_obs"]
    )

    print("\n  Informative censoring check")

    if (
        len(censored) > 5
        and censored["early_rate"].std() > 0
    ):

        rho, p = spearmanr(
            censored["early_rate"],
            censored["last_obs"]
        )

        flag = (
            "<-- WARNING: possible informative censoring"
            if p < 0.05
            else "(clean)"
        )

        print(
            f"    n={len(censored)}, "
            f"rho={rho:.3f}, "
            f"p={p:.4f}  {flag}"
        )

    else:

        print(
            "    Skipped (insufficient variation)."
        )


# ============================================================
# PREPARE COX DATA
# ============================================================

def prepare_cox_data(cohort):

    columns = [
        "duration",
        "event"
    ] + COX_VARIABLES

    sub = cohort[columns].copy()

    for col in columns:

        sub[col] = pd.to_numeric(
            sub[col],
            errors="coerce"
        )

    before = len(sub)

    sub = sub.dropna().copy()

    removed = before - len(sub)

    if removed > 0:

        print(
            "    Rows removed because of missing "
            f"Cox variables: {removed}"
        )

    return sub


# ============================================================
# STANDARDIZE PREDICTORS
# ============================================================

def standardize_predictors(
    data,
    variables=COX_VARIABLES
):

    """
    Returns (standardized_df, usable_variables).

    Near-zero-variance columns are EXCLUDED from the returned
    variable list rather than zero-filled.
    """

    sub = data.copy()

    usable = []

    for col in variables:

        mean = sub[col].mean()
        std = sub[col].std()

        if (
            pd.isna(std)
            or std < MIN_STD_THRESHOLD
        ):

            print(
                f"    WARNING: {col} has near-zero variance "
                f"(std={std:.3g}) -- EXCLUDED from model"
            )

        else:

            sub[col] = (
                sub[col] - mean
            ) / std

            usable.append(col)

    return sub, usable


# ============================================================
# PRINT COX RESULT TABLE
# ============================================================

def print_cox_results(
    cph,
    variables,
    title
):

    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)

    print(
        f"{'Variable':45s}"
        f"{'HR':>10s}"
        f"{'95% CI':>22s}"
        f"{'p-value':>12s}"
    )

    print("-" * 100)

    for variable in variables:

        if variable not in cph.summary.index:
            continue

        row = cph.summary.loc[variable]

        hr = row["exp(coef)"]

        low = row[
            "exp(coef) lower 95%"
        ]

        high = row[
            "exp(coef) upper 95%"
        ]

        p = row["p"]

        print(
            f"{variable:45s}"
            f"{hr:10.3f}"
            f"{low:10.3f} - {high:<10.3f}"
            f"{p:12.4f}"
        )


# ============================================================
# FIT FULL MODEL 4
# ============================================================

def fit_full_model_4(
    cohort,
    name
):

    print("\n" + "=" * 80)
    print(f"COX MODEL 4 -- {name}")
    print("=" * 80)

    sub = prepare_cox_data(cohort)

    sub, usable_vars = standardize_predictors(
        sub,
        COX_VARIABLES
    )

    n = len(sub)

    events = int(
        sub["event"].sum()
    )

    epv = events / max(
        len(usable_vars),
        1
    )

    print(
        f"N used in Cox = {n}"
    )

    print(
        f"Events         = {events}"
    )

    print(
        f"Censored       = {n - events}"
    )

    print(
        f"Variables used = {len(usable_vars)} / "
        f"{len(COX_VARIABLES)} requested"
    )

    print(
        f"EPV            = {epv:.2f}"
    )

    if epv < MIN_EPV:

        print("\n  WARNING:")

        print(
            f"  EPV = {epv:.2f}, below the reference "
            f"threshold of {MIN_EPV}."
        )

        print(
            "  This model is exploratory. Coefficients, "
            "CIs and p-values should"
        )

        print(
            "  NOT be interpreted as definitive "
            "independent effects."
        )

    if len(usable_vars) == 0:

        print(
            "  No usable variables -- cannot fit model."
        )

        return sub, None

    cph = CoxPHFitter(
        penalizer=PENALIZER
    )

    cph.fit(
        sub[
            ["duration", "event"]
            + usable_vars
        ],
        duration_col="duration",
        event_col="event"
    )

    print_cox_results(
        cph,
        usable_vars,
        "FULL MODEL 4 RESULTS"
    )

    print(
        f"\n  C-index = "
        f"{cph.concordance_index_:.3f}"
    )

    print(
        f"  Partial log-likelihood = "
        f"{cph.log_likelihood_:.3f}"
    )

    return sub, cph


# ============================================================
# REDUCED CURRENT-DENSITY SENSITIVITY MODEL
# ============================================================

def fit_reduced_current_model(
    cohort,
    name
):

    variable = COL_CURRENT

    print("\n" + "-" * 80)
    print(
        f"REDUCED SENSITIVITY MODEL -- {name}"
    )
    print("-" * 80)

    columns = [
        "duration",
        "event",
        variable
    ]

    sub = cohort[columns].copy()

    for col in columns:

        sub[col] = pd.to_numeric(
            sub[col],
            errors="coerce"
        )

    sub = sub.dropna().copy()

    std = sub[variable].std()

    if (
        pd.isna(std)
        or std < MIN_STD_THRESHOLD
    ):

        print(
            "  Current density has insufficient variance. "
            "Model skipped."
        )

        return None, sub

    sub[variable] = (
        sub[variable]
        - sub[variable].mean()
    ) / std

    cph = CoxPHFitter(
        penalizer=PENALIZER
    )

    cph.fit(
        sub,
        duration_col="duration",
        event_col="event"
    )

    row = cph.summary.loc[variable]

    print(
        f"  Stress current density:"
    )

    print(
        f"    HR = "
        f"{row['exp(coef)']:.3f}"
    )

    print(
        f"    95% CI = "
        f"[{row['exp(coef) lower 95%']:.3f}, "
        f"{row['exp(coef) upper 95%']:.3f}]"
    )

    print(
        f"    p = "
        f"{row['p']:.4f}"
    )

    print(
        f"    C-index = "
        f"{cph.concordance_index_:.3f}"
    )

    return cph, sub


# ============================================================
# WAFER DIAGNOSTICS
# ============================================================

def wafer_diagnostics(
    cph,
    sub,
    cohort,
    variables_used
):

    if (
        cph is None
        or len(variables_used) == 0
    ):

        print(
            "\n  Wafer diagnostics skipped."
        )

        return

    print("\n" + "-" * 80)
    print("WAFER DIAGNOSTICS")
    print("-" * 80)

    ct = pd.crosstab(
        cohort[COL_WAFER],
        cohort["event"]
    )

    chi2_stat, p_chi2, _, _ = chi2_contingency(ct)

    print(
        f"\n  Wafer vs failure "
        f"(unadjusted chi-square):"
    )

    print(
        f"    chi2 = {chi2_stat:.3f}"
    )

    print(
        f"    p    = {p_chi2:.4f}"
    )

    sub2 = sub.copy()

    sub2[COL_WAFER] = (
        cohort.loc[
            sub2.index,
            COL_WAFER
        ].astype(str).values
    )

    wafer_dummies = pd.get_dummies(
        sub2[COL_WAFER],
        prefix="wafer",
        drop_first=True
    ).astype(float)

    if wafer_dummies.shape[1] == 0:

        print(
            "\n  Only one wafer present."
        )

        return

    sub2 = pd.concat(
        [
            sub2,
            wafer_dummies
        ],
        axis=1
    )

    wafer_cols = list(
        wafer_dummies.columns
    )

    cph_base = CoxPHFitter(
        penalizer=PENALIZER
    )

    cph_base.fit(
        sub2[
            ["duration", "event"]
            + variables_used
        ],
        duration_col="duration",
        event_col="event"
    )

    cph_wafer = CoxPHFitter(
        penalizer=PENALIZER
    )

    cph_wafer.fit(
        sub2[
            ["duration", "event"]
            + variables_used
            + wafer_cols
        ],
        duration_col="duration",
        event_col="event"
    )

    lr = 2 * (
        cph_wafer.log_likelihood_
        - cph_base.log_likelihood_
    )

    p_lr = (
        chi2dist.sf(
            lr,
            len(wafer_cols)
        )
        if lr > 0
        else np.nan
    )

    print(
        f"\n  Wafer dummy-variable LR test:"
    )

    print(
        f"    chi2 = {lr:.3f}"
    )

    print(
        f"    df   = {len(wafer_cols)}"
    )

    print(
        f"    p    = {p_lr:.4f}"
    )

    if p_lr < 0.05:

        print(
            "    Interpretation: possible wafer effect; "
            "inference limited by few failures."
        )

    else:

        print(
            "    Interpretation: no significant evidence "
            "that wafer improves the model."
        )

    try:

        residuals = cph_base.compute_residuals(
            sub2[
                ["duration", "event"]
                + variables_used
            ],
            kind="martingale"
        )

        sub2["martingale"] = (
            residuals["martingale"]
            .reindex(sub2.index)
        )

        residual_data = sub2.dropna(
            subset=[
                "martingale",
                COL_WAFER
            ]
        )

        groups = [
            g["martingale"].values
            for _, g in residual_data.groupby(
                COL_WAFER
            )
        ]

        if (
            len(groups) >= 2
            and any(
                len(g) > 1
                and np.std(g) > 0
                for g in groups
            )
        ):

            h, p_kw = kruskal(
                *groups
            )

            print(
                f"\n  Martingale residual "
                f"Kruskal-Wallis test:"
            )

            print(
                f"    H = {h:.3f}"
            )

            print(
                f"    p = {p_kw:.4f}"
            )

            if p_kw < 0.05:

                print(
                    "    Interpretation: residual differences "
                    "between wafers detected."
                )

                print(
                    "    NOT proof of an independent wafer "
                    "effect given few failures."
                )

            else:

                print(
                    "    Interpretation: no strong residual "
                    "wafer pattern detected."
                )

            print(
                "\n  Martingale residuals by wafer:"
            )

            print(
                residual_data.groupby(
                    COL_WAFER
                )["martingale"]
                .agg(
                    [
                        "mean",
                        "median",
                        "std",
                        "count"
                    ]
                )
                .to_string()
            )

        else:

            print(
                "\n  Martingale residual test skipped "
                "(insufficient variation)."
            )

    except Exception as e:

        print(
            "\n  Martingale residual test could "
            f"not be calculated: {e}"
        )


# ============================================================
# FINAL COMPACT RESULTS TABLE
# ============================================================

def create_results_table(
    cph_broad,
    cph_strict,
    vars_broad,
    vars_strict
):

    common_vars = [
        v
        for v in COX_VARIABLES
        if v in vars_broad
        and v in vars_strict
    ]

    rows = []

    for variable in common_vars:

        row_b = cph_broad.summary.loc[
            variable
        ]

        row_s = cph_strict.summary.loc[
            variable
        ]

        rows.append(
            {
                "Variable": variable,

                "Broad_HR":
                    row_b["exp(coef)"],

                "Broad_CI95_low":
                    row_b[
                        "exp(coef) lower 95%"
                    ],

                "Broad_CI95_high":
                    row_b[
                        "exp(coef) upper 95%"
                    ],

                "Broad_p":
                    row_b["p"],

                "Strict_HR":
                    row_s["exp(coef)"],

                "Strict_CI95_low":
                    row_s[
                        "exp(coef) lower 95%"
                    ],

                "Strict_CI95_high":
                    row_s[
                        "exp(coef) upper 95%"
                    ],

                "Strict_p":
                    row_s["p"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FIGURES
# ============================================================

def plot_early_rate(
    broad,
    strict
):

    print("\n" + "=" * 80)
    print(
        "FIGURE 1 -- EARLY ITH DEGRADATION RATE"
    )
    print("=" * 80)

    broad_plot = broad[
        [
            "early_rate",
            "event"
        ]
    ].dropna().copy()

    strict_plot = strict[
        [
            "early_rate",
            "event"
        ]
    ].dropna().copy()

    broad_plot["Outcome"] = np.where(
        broad_plot["event"] == 1,
        "Failed",
        "Censored"
    )

    strict_plot["Outcome"] = np.where(
        strict_plot["event"] == 1,
        "Failed",
        "Censored"
    )

    # 1A: individual observations + median
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True
    )

    for ax, data, title in [
        (
            axes[0],
            broad_plot,
            "Broad cohort"
        ),
        (
            axes[1],
            strict_plot,
            "Strict cohort"
        )
    ]:

        rng = np.random.default_rng(42)

        for i, outcome in enumerate(
            [
                "Censored",
                "Failed"
            ]
        ):

            values = data.loc[
                data["Outcome"] == outcome,
                "early_rate"
            ].values

            if len(values) == 0:
                continue

            jitter = rng.uniform(
                -0.08,
                0.08,
                size=len(values)
            )

            x = (
                np.full(
                    len(values),
                    i
                )
                + jitter
            )

            ax.scatter(
                x,
                values,
                alpha=0.65,
                s=35,
                label=outcome
            )

            ax.hlines(
                np.median(values),
                i - 0.15,
                i + 0.15,
                linewidth=3
            )

        ax.set_xticks(
            [0, 1]
        )

        ax.set_xticklabels(
            [
                "Censored",
                "Failed"
            ]
        )

        ax.set_xlabel(
            "Outcome"
        )

        ax.set_title(
            title
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

        ax.legend()

    axes[0].set_ylabel(
        "Theil-Sen early ITH degradation rate"
    )

    fig.suptitle(
        "Early ITH degradation rate -- individual observations",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    # 1B: boxplot
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True
    )

    for ax, data, title in [
        (
            axes[0],
            broad_plot,
            "Broad cohort"
        ),
        (
            axes[1],
            strict_plot,
            "Strict cohort"
        )
    ]:

        censored = data.loc[
            data["Outcome"] == "Censored",
            "early_rate"
        ].values

        failed = data.loc[
            data["Outcome"] == "Failed",
            "early_rate"
        ].values

        ax.boxplot(
            [
                censored,
                failed
            ],
            labels=[
                "Censored",
                "Failed"
            ],
            showmeans=True
        )

        ax.set_xlabel(
            "Outcome"
        )

        ax.set_title(
            title
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

    axes[0].set_ylabel(
        "Theil-Sen early ITH degradation rate"
    )

    fig.suptitle(
        "Early ITH degradation rate by outcome",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    # 1C: histograms
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True
    )

    axes[0].hist(
        broad_plot.loc[
            broad_plot["Outcome"] == "Censored",
            "early_rate"
        ],
        bins=15,
        alpha=0.6,
        label="Censored"
    )

    axes[0].hist(
        broad_plot.loc[
            broad_plot["Outcome"] == "Failed",
            "early_rate"
        ],
        bins=15,
        alpha=0.6,
        label="Failed"
    )

    axes[0].set_title(
        "Broad cohort"
    )

    axes[0].set_xlabel(
        "Theil-Sen early ITH degradation rate"
    )

    axes[0].set_ylabel(
        "Count"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    axes[1].hist(
        strict_plot.loc[
            strict_plot["Outcome"] == "Censored",
            "early_rate"
        ],
        bins=15,
        alpha=0.6,
        label="Censored"
    )

    axes[1].hist(
        strict_plot.loc[
            strict_plot["Outcome"] == "Failed",
            "early_rate"
        ],
        bins=15,
        alpha=0.6,
        label="Failed"
    )

    axes[1].set_title(
        "Strict cohort"
    )

    axes[1].set_xlabel(
        "Theil-Sen early ITH degradation rate"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    fig.suptitle(
        "Distribution of early ITH degradation rate",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    print(
        "\nEARLY-RATE NUMERICAL DIAGNOSTICS"
    )

    for name, data in [
        ("Broad", broad_plot),
        ("Strict", strict_plot)
    ]:

        censored = data.loc[
            data["Outcome"] == "Censored",
            "early_rate"
        ]

        failed = data.loc[
            data["Outcome"] == "Failed",
            "early_rate"
        ]

        print(f"\n{name}")

        print(
            f"  Censored: n={len(censored)}, "
            f"median={censored.median():.6g}"
        )

        print(
            f"  Failed:   n={len(failed)}, "
            f"median={failed.median():.6g}"
        )

        if (
            len(censored) > 0
            and len(failed) > 0
        ):

            stat, p = mannwhitneyu(
                censored,
                failed,
                alternative="two-sided"
            )

            print(
                f"  Mann-Whitney U = {stat:.3f}"
            )

            print(
                f"  p = {p:.4f}"
            )


def plot_cavity_length(
    broad,
    strict
):

    print("\n" + "=" * 80)
    print(
        "FIGURE 2 -- CAVITY LENGTH"
    )
    print("=" * 80)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True
    )

    for ax, data, title in [
        (
            axes[0],
            broad,
            "Broad cohort"
        ),
        (
            axes[1],
            strict,
            "Strict cohort"
        )
    ]:

        plot_data = data[
            [
                COL_CAVITY,
                "event"
            ]
        ].copy()

        plot_data[COL_CAVITY] = pd.to_numeric(
            plot_data[COL_CAVITY],
            errors="coerce"
        )

        plot_data = plot_data.dropna()

        censored = plot_data.loc[
            plot_data["event"] == 0,
            COL_CAVITY
        ].values

        failed = plot_data.loc[
            plot_data["event"] == 1,
            COL_CAVITY
        ].values

        ax.boxplot(
            [
                censored,
                failed
            ],
            labels=[
                "Censored",
                "Failed"
            ],
            showmeans=True
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Outcome"
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

    axes[0].set_ylabel(
        "Cavity length"
    )

    fig.suptitle(
        "Cavity length by outcome",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    for name, data in [
        ("Broad", broad),
        ("Strict", strict)
    ]:

        plot_data = data[
            [
                COL_CAVITY,
                "event"
            ]
        ].copy()

        plot_data[COL_CAVITY] = pd.to_numeric(
            plot_data[COL_CAVITY],
            errors="coerce"
        )

        plot_data = plot_data.dropna()

        censored = plot_data.loc[
            plot_data["event"] == 0,
            COL_CAVITY
        ]

        failed = plot_data.loc[
            plot_data["event"] == 1,
            COL_CAVITY
        ]

        if (
            len(censored) > 0
            and len(failed) > 0
        ):

            stat, p = mannwhitneyu(
                censored,
                failed,
                alternative="two-sided"
            )

            print(
                f"\n{name} cavity length:"
            )

            print(
                f"  Censored median = "
                f"{censored.median():.3f}"
            )

            print(
                f"  Failed median   = "
                f"{failed.median():.3f}"
            )

            print(
                f"  Mann-Whitney U = "
                f"{stat:.3f}"
            )

            print(
                f"  p = {p:.4f}"
            )


def plot_current_density(
    broad,
    strict
):

    print("\n" + "=" * 80)
    print(
        "FIGURE 3 -- STRESS CURRENT DENSITY"
    )
    print("=" * 80)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True
    )

    for ax, data, title in [
        (
            axes[0],
            broad,
            "Broad cohort"
        ),
        (
            axes[1],
            strict,
            "Strict cohort"
        )
    ]:

        plot_data = data[
            [
                COL_CURRENT,
                "event"
            ]
        ].copy()

        plot_data[COL_CURRENT] = pd.to_numeric(
            plot_data[COL_CURRENT],
            errors="coerce"
        )

        plot_data = plot_data.dropna()

        censored = plot_data.loc[
            plot_data["event"] == 0,
            COL_CURRENT
        ].values

        failed = plot_data.loc[
            plot_data["event"] == 1,
            COL_CURRENT
        ].values

        ax.boxplot(
            [
                censored,
                failed
            ],
            labels=[
                "Censored",
                "Failed"
            ],
            showmeans=True
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Outcome"
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

    axes[0].set_ylabel(
        "Stress current density [kA/cm2]"
    )

    fig.suptitle(
        "Stress current density by outcome",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    for name, data in [
        ("Broad", broad),
        ("Strict", strict)
    ]:

        plot_data = data[
            [
                COL_CURRENT,
                "event"
            ]
        ].copy()

        plot_data[COL_CURRENT] = pd.to_numeric(
            plot_data[COL_CURRENT],
            errors="coerce"
        )

        plot_data = plot_data.dropna()

        censored = plot_data.loc[
            plot_data["event"] == 0,
            COL_CURRENT
        ]

        failed = plot_data.loc[
            plot_data["event"] == 1,
            COL_CURRENT
        ]

        if (
            len(censored) > 0
            and len(failed) > 0
        ):

            stat, p = mannwhitneyu(
                censored,
                failed,
                alternative="two-sided"
            )

            print(
                f"\n{name} stress current density:"
            )

            print(
                f"  Censored median = "
                f"{censored.median():.3f}"
            )

            print(
                f"  Failed median   = "
                f"{failed.median():.3f}"
            )

            print(
                f"  Mann-Whitney U = "
                f"{stat:.3f}"
            )

            print(
                f"  p = {p:.4f}"
            )


# ============================================================
# NEW FIGURE 4
# EARLY ITH DEGRADATION RATE -- FAILED VS CENSORED
# ============================================================

def plot_early_rate_failed_vs_censored(
    broad,
    strict
):

    print("\n" + "=" * 80)
    print(
        "FIGURE 4 -- EARLY ITH DEGRADATION RATE"
    )
    print("=" * 80)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True
    )

    for ax, data, title in [
        (
            axes[0],
            broad,
            "Broad cohort"
        ),
        (
            axes[1],
            strict,
            "Strict cohort"
        )
    ]:

        plot_data = data[
            [
                "early_rate",
                "event"
            ]
        ].copy()

        plot_data["early_rate"] = pd.to_numeric(
            plot_data["early_rate"],
            errors="coerce"
        )

        plot_data = plot_data.dropna()

        censored = plot_data.loc[
            plot_data["event"] == 0,
            "early_rate"
        ].values

        failed = plot_data.loc[
            plot_data["event"] == 1,
            "early_rate"
        ].values

        ax.boxplot(
            [
                censored,
                failed
            ],
            labels=[
                "Censored",
                "Failed"
            ],
            showmeans=True
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Outcome"
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

    axes[0].set_ylabel(
        "Theil-Sen early ITH degradation rate"
    )

    fig.suptitle(
        "Early ITH degradation rate by outcome",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # NUMERICAL DIAGNOSTICS
    # --------------------------------------------------------

    print(
        "\nEARLY ITH DEGRADATION RATE -- "
        "FAILED VS CENSORED"
    )

    for name, data in [
        ("Broad", broad),
        ("Strict", strict)
    ]:

        plot_data = data[
            [
                "early_rate",
                "event"
            ]
        ].copy()

        plot_data["early_rate"] = pd.to_numeric(
            plot_data["early_rate"],
            errors="coerce"
        )

        plot_data = plot_data.dropna()

        censored = plot_data.loc[
            plot_data["event"] == 0,
            "early_rate"
        ]

        failed = plot_data.loc[
            plot_data["event"] == 1,
            "early_rate"
        ]

        print(
            f"\n{name}"
        )

        print(
            f"  Censored n      = {len(censored)}"
        )

        print(
            f"  Failed n        = {len(failed)}"
        )

        print(
            f"  Censored median = "
            f"{censored.median():.6g}"
        )

        print(
            f"  Failed median   = "
            f"{failed.median():.6g}"
        )

        if (
            len(censored) > 0
            and len(failed) > 0
        ):

            stat, p = mannwhitneyu(
                censored,
                failed,
                alternative="two-sided"
            )

            print(
                f"  Mann-Whitney U = "
                f"{stat:.3f}"
            )

            print(
                f"  p = {p:.4f}"
            )


def plot_wafer_martingale_residuals(
    broad,
    strict,
    cph_broad,
    sub_broad,
    cph_strict,
    sub_strict
):

    print("\n" + "=" * 80)
    print(
        "FIGURE 5 -- MARTINGALE RESIDUALS BY WAFER"
    )
    print("=" * 80)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True
    )

    cohorts = [
        (
            "Broad",
            broad,
            cph_broad,
            sub_broad,
            axes[0]
        ),
        (
            "Strict",
            strict,
            cph_strict,
            sub_strict,
            axes[1]
        ),
    ]

    for (
        name,
        cohort,
        cph,
        cox_data,
        ax
    ) in cohorts:

        if cph is None:

            ax.set_title(
                f"{name} cohort -- no model"
            )

            continue

        used_vars = [
            c
            for c in COX_VARIABLES
            if (
                c in cox_data.columns
                and c not in (
                    "duration",
                    "event"
                )
            )
        ]

        residuals = cph.compute_residuals(
            cox_data[
                [
                    "duration",
                    "event"
                ] + used_vars
            ],
            kind="martingale"
        )

        residual_data = residuals[
            ["martingale"]
        ].copy()

        residual_data[COL_WAFER] = (
            cohort.loc[
                residual_data.index,
                COL_WAFER
            ].astype(str)
        )

        residual_data = residual_data.dropna(
            subset=[
                "martingale",
                COL_WAFER
            ]
        )

        groups, labels = [], []

        for wafer, group in residual_data.groupby(
            COL_WAFER
        ):

            labels.append(
                str(wafer)
            )

            groups.append(
                group["martingale"].values
            )

        if len(groups) == 0:

            ax.set_title(
                f"{name} cohort -- no residuals"
            )

            continue

        ax.boxplot(
            groups,
            labels=labels,
            showmeans=True
        )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1
        )

        ax.set_title(
            name + " cohort"
        )

        ax.set_xlabel(
            "Wafer"
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

        if (
            len(groups) >= 2
            and any(
                len(g) > 1
                and np.std(g) > 0
                for g in groups
            )
        ):

            h, p_kw = kruskal(
                *groups
            )

            print(
                f"\n{name} martingale residuals:"
            )

            print(
                f"  Kruskal-Wallis H = "
                f"{h:.3f}"
            )

            print(
                f"  p = {p_kw:.4f}"
            )

        else:

            print(
                f"\n{name} martingale residuals:"
            )

            print(
                "  Kruskal-Wallis test skipped."
            )

        print(
            f"\n  {name} residual summary:"
        )

        print(
            residual_data.groupby(
                COL_WAFER
            )["martingale"]
            .agg(
                [
                    "mean",
                    "median",
                    "std",
                    "count"
                ]
            )
            .to_string()
        )

    axes[0].set_ylabel(
        "Martingale residual"
    )

    fig.suptitle(
        "Cox Model 4 martingale residuals by wafer",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


def plot_all_figures(
    broad,
    strict,
    cph_broad,
    sub_broad,
    cph_strict,
    sub_strict
):

    plot_early_rate(
        broad,
        strict
    )

    plot_cavity_length(
        broad,
        strict
    )

    plot_current_density(
        broad,
        strict
    )

    # NEW FIGURE 4
    plot_early_rate_failed_vs_censored(
        broad,
        strict
    )

    # Martingale residuals are now Figure 5
    plot_wafer_martingale_residuals(
        broad,
        strict,
        cph_broad,
        sub_broad,
        cph_strict,
        sub_strict
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df, ith_times = load_data()

    df = add_features(
        df,
        ith_times
    )

    broad_diag, strict_diag = diagnose_cohort_steps(
        df
    )

    broad = build_cohort(
        df,
        strict=False
    )

    strict = build_cohort(
        df,
        strict=True
    )

    print("\n" + "=" * 80)
    print(
        "CONSISTENCY CHECK: "
        "DIAGNOSTIC vs ANALYSIS COHORTS"
    )
    print("=" * 80)

    assert (
        len(broad) == len(broad_diag)
    ), "Broad cohort mismatch."

    assert (
        len(strict) == len(strict_diag)
    ), "Strict cohort mismatch."

    print(
        "OK: diagnostic and analysis cohorts agree."
    )

    print(
        f"Broad:  N={len(broad)}, "
        f"events={int(broad['event'].sum())}"
    )

    print(
        f"Strict: N={len(strict)}, "
        f"events={int(strict['event'].sum())}"
    )

    if len(broad) == len(strict):

        print(
            "\n(Broad == Strict -- expected only if "
            "no ITH readings exist beyond 168h;"
        )

        print(
            " check the auto-discovered timepoint "
            "list printed at load time.)"
        )

    print("\n" + "=" * 80)
    print("COHORT SUMMARY")
    print("=" * 80)

    for name, cohort in [
        ("Broad", broad),
        ("Strict", strict)
    ]:

        n = len(cohort)

        events = int(
            cohort["event"].sum()
        )

        print(
            f"\n{name}"
        )

        print(
            f"  N          = {n}"
        )

        print(
            f"  Failures   = {events}"
        )

        print(
            f"  Censored   = {n - events}"
        )

        if n > 0:

            print(
                f"  Failure %  = "
                f"{100 * events / n:.2f}%"
            )

        print(
            f"  Median duration = "
            f"{cohort['duration'].median():.1f} h"
        )

    # ========================================================
    # BROAD
    # ========================================================

    print("\n" + "#" * 80)
    print("BROAD COHORT ANALYSIS")
    print("#" * 80)

    informative_censoring_check(
        broad
    )

    sub_broad, cph_broad = fit_full_model_4(
        broad,
        "Broad 168h"
    )

    _, _ = fit_reduced_current_model(
        broad,
        "Broad 168h"
    )

    vars_broad = [
        c
        for c in COX_VARIABLES
        if (
            cph_broad is not None
            and c in cph_broad.summary.index
        )
    ]

    wafer_diagnostics(
        cph_broad,
        sub_broad,
        broad,
        vars_broad
    )

    # ========================================================
    # STRICT
    # ========================================================

    print("\n" + "#" * 80)
    print("STRICT COHORT ANALYSIS")
    print("#" * 80)

    informative_censoring_check(
        strict
    )

    sub_strict, cph_strict = fit_full_model_4(
        strict,
        "Strict 168h"
    )

    _, _ = fit_reduced_current_model(
        strict,
        "Strict 168h"
    )

    vars_strict = [
        c
        for c in COX_VARIABLES
        if (
            cph_strict is not None
            and c in cph_strict.summary.index
        )
    ]

    wafer_diagnostics(
        cph_strict,
        sub_strict,
        strict,
        vars_strict
    )

    # ========================================================
    # COMPACT RESULTS TABLE
    # ========================================================

    if (
        cph_broad is not None
        and cph_strict is not None
    ):

        print("\n" + "=" * 120)

        print(
            "COX MODEL 4 -- COMPACT RESULTS "
            "(variables present in BOTH models)"
        )

        print("=" * 120)

        final_table = create_results_table(
            cph_broad,
            cph_strict,
            vars_broad,
            vars_strict
        )

        display_table = final_table.copy()

        for col in [
            "Broad_HR",
            "Broad_CI95_low",
            "Broad_CI95_high",
            "Broad_p",
            "Strict_HR",
            "Strict_CI95_low",
            "Strict_CI95_high",
            "Strict_p"
        ]:

            display_table[col] = (
                display_table[col]
                .astype(float)
                .map(
                    lambda x: f"{x:.4f}"
                )
            )

        print(
            display_table.to_string(
                index=False
            )
        )

    # ========================================================
    # MODEL-LEVEL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("MODEL-LEVEL SUMMARY")
    print("=" * 80)

    if cph_broad is not None:

        print(
            f"\nBroad:"
        )

        print(
            f"  N used in Model 4 = "
            f"{len(sub_broad)}"
        )

        print(
            f"  Events             = "
            f"{int(sub_broad['event'].sum())}"
        )

        print(
            f"  Variables used     = "
            f"{len(vars_broad)}"
        )

        print(
            f"  EPV                = "
            f"{int(sub_broad['event'].sum()) / max(len(vars_broad),1):.2f}"
        )

        print(
            f"  C-index            = "
            f"{cph_broad.concordance_index_:.3f}"
        )

    if cph_strict is not None:

        print(
            f"\nStrict:"
        )

        print(
            f"  N used in Model 4 = "
            f"{len(sub_strict)}"
        )

        print(
            f"  Events             = "
            f"{int(sub_strict['event'].sum())}"
        )

        print(
            f"  Variables used     = "
            f"{len(vars_strict)}"
        )

        print(
            f"  EPV                = "
            f"{int(sub_strict['event'].sum()) / max(len(vars_strict),1):.2f}"
        )

        print(
            f"  C-index            = "
            f"{cph_strict.concordance_index_:.3f}"
        )

    # ========================================================
    # INTERPRETATION NOTE
    # ========================================================

    print("\n" + "=" * 80)
    print("INTERPRETATION NOTE")
    print("=" * 80)

    print(
        "\nThe full Model 4 is retained for exploratory comparison."
    )

    print(
        "With few events, EPV may be low -- coefficients, CIs "
        "and p-values from"
    )

    print(
        "the full model should be read cautiously, not as "
        "definitive independent"
    )

    print(
        "effects. The reduced current-density-only model is "
        "the more trustworthy"
    )

    print(
        "reference/sensitivity result at low EPV."
    )

    # ========================================================
    # FIGURES
    # ========================================================

    plot_all_figures(
        broad,
        strict,
        cph_broad,
        sub_broad,
        cph_strict,
        sub_strict
    )
# ============================================================
# EARLY_RATE STANDALONE PREDICTIVE VALIDATION
# Add this near the bottom of your script, after cohorts are built.
# ============================================================
#this to make sure that the Theil-Sen regression results are accurate 
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd


def early_rate_predictive_validation(cohort, name, time_horizon=1000):
    """
    Standalone evaluation of early_rate as a sole predictor of failure.
    Reports:
      - Univariate Cox HR/CI/p for early_rate alone
      - C-index (in-sample) for early_rate alone
      - Leave-one-out C-index (accounts for tiny event count)
      - Operational threshold: KM failure risk by time_horizon,
        stratified by early_rate above/below median (and a few
        alternative cutoffs) with counts and events per stratum
    """

    print("\n" + "=" * 80)
    print(f"EARLY_RATE STANDALONE VALIDATION -- {name}")
    print("=" * 80)

    data = cohort[["duration", "event", "early_rate"]].copy()
    data["early_rate"] = pd.to_numeric(data["early_rate"], errors="coerce")
    data = data.dropna()

    n = len(data)
    events = int(data["event"].sum())

    print(f"N = {n}, events = {events}")

    if events < 3:
        print("  Too few events to validate reliably. Skipping.")
        return

    # ------------------------------------------------------------
    # 1. Univariate Cox model: early_rate alone
    # ------------------------------------------------------------
    std = data["early_rate"].std()
    if pd.isna(std) or std < 1e-12:
        print("  early_rate has no variance. Skipping.")
        return

    data_std = data.copy()
    data_std["early_rate_z"] = (
        data_std["early_rate"] - data_std["early_rate"].mean()
    ) / std

    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(
        data_std[["duration", "event", "early_rate_z"]],
        duration_col="duration",
        event_col="event",
    )

    row = cph.summary.loc["early_rate_z"]
    print("\n  Univariate Cox (early_rate only):")
    print(f"    HR    = {row['exp(coef)']:.3f}")
    print(
        f"    95% CI= [{row['exp(coef) lower 95%']:.3f}, "
        f"{row['exp(coef) upper 95%']:.3f}]"
    )
    print(f"    p     = {row['p']:.4f}")
    print(f"    In-sample C-index = {cph.concordance_index_:.3f}")

    # ------------------------------------------------------------
    # 2. Leave-one-out C-index
    # ------------------------------------------------------------
    # For each held-out point, fit Cox on the remaining N-1,
    # predict partial hazard for the held-out point, then compute
    # concordance across all held-out predictions vs actual outcomes.
    loo_risk_scores = np.full(n, np.nan)
    idx_list = data_std.index.to_list()

    for i, hold_idx in enumerate(idx_list):
        train = data_std.drop(index=hold_idx)
        test = data_std.loc[[hold_idx]]

        if train["event"].sum() < 2:
            continue  # cannot fit reliably

        try:
            cph_loo = CoxPHFitter(penalizer=0.05)
            cph_loo.fit(
                train[["duration", "event", "early_rate_z"]],
                duration_col="duration",
                event_col="event",
            )
            risk = cph_loo.predict_partial_hazard(test).values[0]
            loo_risk_scores[i] = risk
        except Exception:
            continue

    valid = ~np.isnan(loo_risk_scores)

    if valid.sum() >= 5 and data_std["event"].values[valid].sum() >= 2:
        loo_c = concordance_index(
            data_std["duration"].values[valid],
            -loo_risk_scores[valid],  # higher risk -> shorter duration
            data_std["event"].values[valid],
        )
        print(f"\n  Leave-one-out C-index = {loo_c:.3f}  (n_valid={valid.sum()})")
    else:
        print("\n  Leave-one-out C-index: insufficient valid folds to compute.")

    # ------------------------------------------------------------
    # 3. Operational threshold: KM failure risk by time_horizon
    #    stratified by early_rate cutoffs
    # ------------------------------------------------------------
    print(f"\n  Operational threshold analysis (horizon = {time_horizon} h):")

    cutoffs = {
        "median": data["early_rate"].median(),
        "p75": data["early_rate"].quantile(0.75),
        "zero": 0.0,
    }

    for label, cutoff in cutoffs.items():
        high = data[data["early_rate"] > cutoff]
        low = data[data["early_rate"] <= cutoff]

        print(f"\n    Cutoff = {label} (early_rate > {cutoff:.6g}):")
        print(f"      High group: n={len(high)}, events={int(high['event'].sum())}")
        print(f"      Low group:  n={len(low)}, events={int(low['event'].sum())}")

        for grp_name, grp in [("High", high), ("Low", low)]:
            if len(grp) < 3 or grp["event"].sum() == 0:
                print(f"        {grp_name}: insufficient data for KM estimate.")
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(grp["duration"], event_observed=grp["event"])

            try:
                surv_at_t = kmf.survival_function_at_times(time_horizon).values[0]
                risk_at_t = 1 - surv_at_t
                print(
                    f"        {grp_name}: estimated failure risk by "
                    f"{time_horizon}h = {risk_at_t*100:.1f}%"
                )
            except Exception:
                print(f"        {grp_name}: could not evaluate at horizon.")

    print(
        "\n  NOTE: with n_events =", events,
        "these strata estimates carry wide uncertainty.",
        "Report alongside CIs, not as point estimates alone."
    )


# ------------------------------------------------------------
# Run for both cohorts (call this after broad/strict are built)
# ------------------------------------------------------------
early_rate_predictive_validation(broad, "Broad 168h", time_horizon=1000)
early_rate_predictive_validation(strict, "Strict 168h", time_horizon=1000)