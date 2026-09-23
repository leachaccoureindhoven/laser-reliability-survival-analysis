# -*- coding: utf-8 -*-
"""
EARLY ITH DEGRADATION — 168 h LANDMARK SURVIVAL VALIDATION

Main question:
    Are devices with faster early ITH degradation more likely to fail later?

Analysis:
    1. Theil–Sen early degradation rate
    2. Broad + Strict 168 h landmark cohorts
    3. Univariate Cox proportional-hazards model
    4. Kaplan–Meier survival comparison
    5. Top-25% failure-concentration analysis

Landmark definition:
    - Early rate uses valid ITH measurements <= 168 h
    - Minimum 3 early measurements
    - Failures <= 168 h are excluded
    - All follow-up after 168 h is retained
    - Failure time = actual failure milestone
    - Censoring time = actual last observation

@author: Léa Chaccour
"""

# =============================================================================
# IMPORTS
# =============================================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import theilslopes
from lifelines import CoxPHFitter, KaplanMeierFitter


# =============================================================================
# PUBLICATION-QUALITY FIGURE SETTINGS
# =============================================================================

plt.rcParams.update({

    # Font
    "font.family": "Times New Roman",
    "font.size": 22,
    "font.weight": "bold",

    # Axes
    "axes.titlesize": 22,
    "axes.titleweight": "bold",

    "axes.labelsize": 22,
    "axes.labelweight": "bold",

    # Tick labels
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,

    # Legend
    "legend.fontsize": 22,

    # Figure title
    "figure.titlesize": 26,
    "figure.titleweight": "bold",

    # Math text
    "mathtext.fontset": "stix",
})


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_FILE = (
    r".....csv"
)

WINDOW_H = 168
MIN_EARLY_POINTS = 3

# Small penalizer for numerical stability of the Cox model.
PENALIZER = 0.05

RANDOM_SEED = 0


# =============================================================================
# COLUMN NAMES
# =============================================================================

COL_FAIL = "FAIL_ITH0"
COL_FAIL_TIME = "FAIL_ITH0_milestone"
COL_ITH0 = "ITH_0"


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_data():

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset shape: {df.shape}")

    return df


# =============================================================================
# 2. DISCOVER ITH TIMEPOINT COLUMNS
# =============================================================================

def discover_ith_columns(df):

    ith_columns = {}

    for col in df.columns:

        match = re.match(
            r"ITH_(\d+(?:\.\d+)?)$",
            str(col)
        )

        if match:

            time_h = float(match.group(1))

            if time_h.is_integer():
                time_h = int(time_h)

            ith_columns[time_h] = col

    ith_columns = dict(
        sorted(ith_columns.items())
    )

    print("\nAuto-discovered ITH timepoint columns:")
    print(list(ith_columns.keys()))

    print(
        f"Number of ITH timepoints: "
        f"{len(ith_columns)}"
    )

    return ith_columns


# =============================================================================
# 3. CALCULATE THEIL–SEN EARLY RATE
# =============================================================================

def calculate_early_rate(
    row,
    ith_columns
):
    """
    Calculate robust early ITH degradation rate using
    all valid ITH measurements collected at <=168 h.

    Minimum number of measurements:
        MIN_EARLY_POINTS = 3
    """

    times = []
    values = []

    for time_h, col in ith_columns.items():

        if time_h > WINDOW_H:
            break

        value = pd.to_numeric(
            row[col],
            errors="coerce"
        )

        if pd.notna(value):

            times.append(
                float(time_h)
            )

            values.append(
                float(value)
            )

    if len(times) < MIN_EARLY_POINTS:
        return np.nan

    times = np.asarray(times)
    values = np.asarray(values)

    try:

        slope, intercept, low_slope, high_slope = theilslopes(
            values,
            times,
            alpha=0.95
        )

        return float(slope)

    except Exception:

        return np.nan


# =============================================================================
# 4. CALCULATE LAST OBSERVATION
# =============================================================================

def calculate_last_observation(
    row,
    ith_columns
):
    """
    Find the last timepoint at which a valid ITH measurement exists.
    """

    observed_times = []

    for time_h, col in ith_columns.items():

        value = pd.to_numeric(
            row[col],
            errors="coerce"
        )

        if pd.notna(value):

            observed_times.append(
                float(time_h)
            )

    if len(observed_times) == 0:
        return np.nan

    return max(observed_times)


# =============================================================================
# 5. ADD EARLY RATE + LAST OBSERVATION
# =============================================================================

def add_features(df):

    print("\n" + "=" * 80)
    print("CALCULATING EARLY DEGRADATION FEATURES")
    print("=" * 80)

    df = df.copy()

    ith_columns = discover_ith_columns(df)

    # -------------------------------------------------------------------------
    # Theil–Sen early degradation rate
    # -------------------------------------------------------------------------

    print(
        "\nCalculating Theil–Sen early degradation rate..."
    )

    df["early_rate"] = df.apply(
        lambda row:
        calculate_early_rate(
            row,
            ith_columns
        ),
        axis=1
    )

    print(
        f"Valid early rates: "
        f"{df['early_rate'].notna().sum()} / {len(df)}"
    )

    # -------------------------------------------------------------------------
    # Last observation
    # -------------------------------------------------------------------------

    print(
        "\nCalculating last observation time..."
    )

    df["last_obs"] = df.apply(
        lambda row:
        calculate_last_observation(
            row,
            ith_columns
        ),
        axis=1
    )

    print(
        f"Valid last observations: "
        f"{df['last_obs'].notna().sum()} / {len(df)}"
    )

    return df, ith_columns


# =============================================================================
# 6. DIAGNOSTIC COHORT STEPS
# =============================================================================

def diagnose_cohort_steps(df):

    print("\n" + "=" * 80)
    print("COHORT DIAGNOSTICS")
    print("=" * 80)

    sub = df.copy()

    print(
        f"\nInitial dataset: {len(sub)}"
    )

    # -------------------------------------------------------------------------
    # Valid ITH_0
    # -------------------------------------------------------------------------

    ith0 = pd.to_numeric(
        sub[COL_ITH0],
        errors="coerce"
    )

    mask_ith0 = ith0.notna()

    sub = sub.loc[
        mask_ith0
    ].copy()

    print(
        f"Valid ITH_0: {len(sub)} "
        f"(removed {mask_ith0.size - mask_ith0.sum()})"
    )

    # -------------------------------------------------------------------------
    # Valid early rate
    # -------------------------------------------------------------------------

    mask_early = sub[
        "early_rate"
    ].notna()

    sub = sub.loc[
        mask_early
    ].copy()

    print(
        f"Valid early rate "
        f"(>= {MIN_EARLY_POINTS} early points): "
        f"{len(sub)}"
    )

    # -------------------------------------------------------------------------
    # Failure information
    # -------------------------------------------------------------------------

    fail = pd.to_numeric(
        sub[COL_FAIL],
        errors="coerce"
    )

    fail_time = pd.to_numeric(
        sub[COL_FAIL_TIME],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # Remove failures <= landmark
    # -------------------------------------------------------------------------

    early_failure = (
        fail.eq(1)
        &
        fail_time.notna()
        &
        fail_time.le(WINDOW_H)
    )

    print(
        f"Failures <= {WINDOW_H} h: "
        f"{early_failure.sum()}"
    )

    sub = sub.loc[
        ~early_failure
    ].copy()

    # -------------------------------------------------------------------------
    # Require follow-up to reach landmark
    # -------------------------------------------------------------------------

    sub["last_obs"] = pd.to_numeric(
        sub["last_obs"],
        errors="coerce"
    )

    valid_landmark_followup = (
        sub["last_obs"].ge(WINDOW_H)
    )

    sub = sub.loc[
        valid_landmark_followup
    ].copy()

    print(
        f"Final Broad cohort: {len(sub)}"
    )

    # -------------------------------------------------------------------------
    # Event definition
    # -------------------------------------------------------------------------

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
        &
        fail_time.notna()
        &
        fail_time.gt(WINDOW_H)
    ).astype(int)

    # -------------------------------------------------------------------------
    # Duration
    # -------------------------------------------------------------------------

    sub["duration"] = np.where(
        sub["event"].eq(1),
        fail_time,
        sub["last_obs"]
    )

    print(
        f"Events after {WINDOW_H} h: "
        f"{sub['event'].sum()}"
    )

    print(
        f"Maximum follow-up: "
        f"{sub['duration'].max():.0f} h"
    )

    return sub


# =============================================================================
# 7. BUILD BROAD / STRICT COHORT
# =============================================================================

def build_cohort(
    df,
    strict=False
):
    """
    Broad:
        - valid ITH_0
        - >=3 early ITH measurements
        - follow-up >=168 h
        - no failure <=168 h

    Strict:
        - same as Broad
        - exact ITH_168 must be available
    """

    sub = df.copy()

    # -------------------------------------------------------------------------
    # Valid ITH_0
    # -------------------------------------------------------------------------

    ith0 = pd.to_numeric(
        sub[COL_ITH0],
        errors="coerce"
    )

    sub = sub.loc[
        ith0.notna()
    ].copy()

    # -------------------------------------------------------------------------
    # Valid early rate
    # -------------------------------------------------------------------------

    sub = sub.loc[
        sub["early_rate"].notna()
    ].copy()

    # -------------------------------------------------------------------------
    # Landmark follow-up
    # -------------------------------------------------------------------------

    sub["last_obs"] = pd.to_numeric(
        sub["last_obs"],
        errors="coerce"
    )

    sub = sub.loc[
        sub["last_obs"].ge(WINDOW_H)
    ].copy()

    # -------------------------------------------------------------------------
    # Failure information
    # -------------------------------------------------------------------------

    fail = pd.to_numeric(
        sub[COL_FAIL],
        errors="coerce"
    )

    fail_time = pd.to_numeric(
        sub[COL_FAIL_TIME],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # Remove failures <=168 h
    # -------------------------------------------------------------------------

    early_failure = (
        fail.eq(1)
        &
        fail_time.notna()
        &
        fail_time.le(WINDOW_H)
    )

    sub = sub.loc[
        ~early_failure
    ].copy()

    # -------------------------------------------------------------------------
    # Strict cohort
    # -------------------------------------------------------------------------

    if strict:

        if "ITH_168" not in sub.columns:

            raise ValueError(
                "Strict cohort requested but "
                "ITH_168 is not available."
            )

        ith168 = pd.to_numeric(
            sub["ITH_168"],
            errors="coerce"
        )

        sub = sub.loc[
            ith168.notna()
        ].copy()

    # -------------------------------------------------------------------------
    # Event
    # -------------------------------------------------------------------------

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
        &
        fail_time.notna()
        &
        fail_time.gt(WINDOW_H)
    ).astype(int)

    # -------------------------------------------------------------------------
    # Duration
    # -------------------------------------------------------------------------

    sub["duration"] = np.where(
        sub["event"].eq(1),
        fail_time,
        sub["last_obs"]
    )

    sub["duration"] = pd.to_numeric(
        sub["duration"],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # Final validity
    # -------------------------------------------------------------------------

    sub = sub.loc[
        sub["duration"].notna()
        &
        sub["duration"].ge(WINDOW_H)
    ].copy()

    return sub


# =============================================================================
# 8. COHORT SUMMARY
# =============================================================================

def print_cohort_summary(
    cohort,
    name
):

    print("\n" + "=" * 80)
    print(f"{name.upper()} COHORT")
    print("=" * 80)

    n = len(cohort)

    events = int(
        cohort["event"].sum()
    )

    censored = n - events

    print(
        f"N                  = {n}"
    )

    print(
        f"Events             = {events}"
    )

    print(
        f"Censored           = {censored}"
    )

    print(
        f"Maximum follow-up  = "
        f"{cohort['duration'].max():.0f} h"
    )


# =============================================================================
# 9. PREPARE COX DATA
# =============================================================================

def prepare_cox_data(
    cohort
):

    cox_data = cohort[
        [
            "duration",
            "event",
            "early_rate"
        ]
    ].copy()

    # Standardize early rate.
    # HR therefore corresponds to a 1-SD increase.

    mean_rate = cox_data[
        "early_rate"
    ].mean()

    sd_rate = cox_data[
        "early_rate"
    ].std()

    if (
        not np.isfinite(sd_rate)
        or
        sd_rate == 0
    ):

        raise ValueError(
            "Early-rate standard deviation "
            "is zero or invalid."
        )

    cox_data[
        "early_rate_z"
    ] = (
        cox_data["early_rate"]
        - mean_rate
    ) / sd_rate

    cox_data = cox_data[
        [
            "duration",
            "event",
            "early_rate_z"
        ]
    ].dropna()

    return cox_data


# =============================================================================
# 10. UNIVARIATE COX
# =============================================================================

def fit_univariate_cox(
    cohort,
    name
):

    print("\n" + "=" * 80)
    print(f"UNIVARIATE COX — {name}")
    print("=" * 80)

    cox_data = prepare_cox_data(
        cohort
    )

    print(
        f"N = {len(cox_data)}, "
        f"events = "
        f"{int(cox_data['event'].sum())}"
    )

    cph = CoxPHFitter(
        penalizer=PENALIZER
    )

    cph.fit(
        cox_data,
        duration_col="duration",
        event_col="event"
    )

    result = cph.summary.loc[
        "early_rate_z"
    ]

    hr = result[
        "exp(coef)"
    ]

    ci_low = result[
        "exp(coef) lower 95%"
    ]

    ci_high = result[
        "exp(coef) upper 95%"
    ]

    p = result[
        "p"
    ]

    print(
        "\nEarly degradation rate "
        "(per 1 SD increase):"
    )

    print(
        f"  HR       = {hr:.3f}"
    )

    print(
        f"  95% CI   = "
        f"[{ci_low:.3f}, {ci_high:.3f}]"
    )

    print(
        f"  p        = {p:.4g}"
    )

    return {
        "Cohort": name,
        "N": len(cox_data),
        "Events": int(
            cox_data["event"].sum()
        ),
        "HR": hr,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "p": p
    }


# =============================================================================
# 11. KAPLAN–MEIER
# =============================================================================

def plot_kaplan_meier(
    cohort,
    name
):
    """
    Compare:
        Top 25% fastest early degradation
        Bottom 75% slower early degradation

    Early degradation is defined using measurements up to 168 h.

    After the 168 h landmark, each device is followed until failure
    or its last available observation.
    """

    sub = cohort.copy()

    q75 = sub[
        "early_rate"
    ].quantile(0.75)

    top25 = sub[
        sub["early_rate"] >= q75
    ].copy()

    bottom75 = sub[
        sub["early_rate"] < q75
    ].copy()

    km_top = KaplanMeierFitter()
    km_bottom = KaplanMeierFitter()

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # -------------------------------------------------------------------------
    # Top 25%
    # -------------------------------------------------------------------------

    km_top.fit(
        top25["duration"],
        event_observed=top25["event"],
        label=(
            f"Top 25% fastest "
            f"(N={len(top25)}, "
            f"events={int(top25['event'].sum())})"
        )
    )

    km_top.plot_survival_function(
        ax=ax,
        ci_show=True
    )

    # -------------------------------------------------------------------------
    # Bottom 75%
    # -------------------------------------------------------------------------

    km_bottom.fit(
        bottom75["duration"],
        event_observed=bottom75["event"],
        label=(
            f"Bottom 75% slower "
            f"(N={len(bottom75)}, "
            f"events={int(bottom75['event'].sum())})"
        )
    )

    km_bottom.plot_survival_function(
        ax=ax,
        ci_show=True
    )

    # -------------------------------------------------------------------------
    # Axis labels
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        "Time (h)",
        fontname="Times New Roman",
        fontsize=22,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Survival probability",
        fontname="Times New Roman",
        fontsize=22,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    ax.set_title(
        "Kaplan–Meier Survival by Early ITH Degradation Rate\n"
        + name,
        fontname="Times New Roman",
        fontsize=26,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Tick labels
    # -------------------------------------------------------------------------

    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(22)
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(22)
        label.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    legend = ax.legend()

    for text in legend.get_texts():
        text.set_fontname("Times New Roman")
        text.set_fontsize(22)
        text.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Grid and limits
    # -------------------------------------------------------------------------

    ax.set_ylim(
        0,
        1.05
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# 12. TOP 25% FAILURE-CONCENTRATION ANALYSIS
# =============================================================================

def top25_analysis(
    cohort,
    name
):

    print("\n" + "=" * 80)
    print(f"TOP 25% ANALYSIS — {name}")
    print("=" * 80)

    sub = cohort.copy()

    # -------------------------------------------------------------------------
    # Define fastest 25%
    # -------------------------------------------------------------------------

    q75 = sub[
        "early_rate"
    ].quantile(0.75)

    top25 = sub[
        sub["early_rate"] >= q75
    ].copy()

    bottom75 = sub[
        sub["early_rate"] < q75
    ].copy()

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    top_events = int(
        top25["event"].sum()
    )

    bottom_events = int(
        bottom75["event"].sum()
    )

    total_events = int(
        sub["event"].sum()
    )

    # -------------------------------------------------------------------------
    # Failure rates
    # -------------------------------------------------------------------------

    top_failure_rate = (
        100 * top25["event"].mean()
        if len(top25) > 0
        else np.nan
    )

    bottom_failure_rate = (
        100 * bottom75["event"].mean()
        if len(bottom75) > 0
        else np.nan
    )

    # -------------------------------------------------------------------------
    # Failure concentration
    # -------------------------------------------------------------------------

    if total_events > 0:

        top_failure_concentration = (
            100
            * top_events
            / total_events
        )

    else:

        top_failure_concentration = np.nan

    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------

    print(
        f"\n75th percentile early rate = "
        f"{q75:.6g}"
    )

    print(
        "\nTop 25% fastest:"
    )

    print(
        f"  N = {len(top25)}"
    )

    print(
        f"  Events = {top_events}"
    )

    print(
        f"  Failure rate = "
        f"{top_failure_rate:.2f}%"
    )

    print(
        "\nBottom 75% slower:"
    )

    print(
        f"  N = {len(bottom75)}"
    )

    print(
        f"  Events = {bottom_events}"
    )

    print(
        f"  Failure rate = "
        f"{bottom_failure_rate:.2f}%"
    )

    print(
        f"\nFailure concentration in top 25% = "
        f"{top_failure_concentration:.1f}% "
        f"({top_events}/{total_events})"
    )

    return {
        "Cohort": name,
        "Top25_N": len(top25),
        "Top25_events": top_events,
        "Top25_failure_rate_%": top_failure_rate,
        "Bottom75_N": len(bottom75),
        "Bottom75_events": bottom_events,
        "Bottom75_failure_rate_%": bottom_failure_rate,
        "Failure_concentration_top25_%":
            top_failure_concentration
    }


# =============================================================================
# 13. FAILURE-CONCENTRATION PLOT
# =============================================================================

def plot_failure_concentration(
    cohort,
    name
):
    """
    Plot what percentage of all failures occur in:
        - Top 25% fastest degraders
        - Bottom 75% slower degraders
    """

    sub = cohort.copy()

    q75 = sub[
        "early_rate"
    ].quantile(0.75)

    top25 = sub[
        sub["early_rate"] >= q75
    ]

    bottom75 = sub[
        sub["early_rate"] < q75
    ]

    top_events = int(
        top25["event"].sum()
    )

    bottom_events = int(
        bottom75["event"].sum()
    )

    total_events = int(
        sub["event"].sum()
    )

    if total_events > 0:

        top_fraction = (
            100
            * top_events
            / total_events
        )

        bottom_fraction = (
            100
            * bottom_events
            / total_events
        )

    else:

        top_fraction = np.nan
        bottom_fraction = np.nan

    categories = [
        "Top 25%\nfastest",
        "Bottom 75%\nslower"
    ]

    values = [
        top_fraction,
        bottom_fraction
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    bars = ax.bar(
        categories,
        values
    )

    # -------------------------------------------------------------------------
    # Percentage labels above bars
    # -------------------------------------------------------------------------

    for bar, value in zip(
        bars,
        values
    ):

        if np.isfinite(value):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontname="Times New Roman",
                fontsize=22,
                fontweight="bold"
            )

    # -------------------------------------------------------------------------
    # Axis labels
    # -------------------------------------------------------------------------

    ax.set_ylabel(
        "Fraction of all failures (%)",
        fontname="Times New Roman",
        fontsize=22,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    ax.set_title(
        "Failure Concentration by Early ITH Degradation Rate\n"
        + name,
        fontname="Times New Roman",
        fontsize=26,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Tick labels
    # -------------------------------------------------------------------------

    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(22)
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(22)
        label.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Limits and grid
    # -------------------------------------------------------------------------

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================

def print_final_summary(
    cox_results,
    top25_results
):

    cox_summary = pd.DataFrame(
        cox_results
    )

    top25_summary = pd.DataFrame(
        top25_results
    )

    summary = cox_summary.merge(
        top25_summary,
        on="Cohort"
    )

    print("\n" + "=" * 80)
    print("FINAL EARLY-DEGRADATION VALIDATION SUMMARY")
    print("=" * 80)

    display_columns = [
        "Cohort",
        "N",
        "Events",
        "HR",
        "CI_low",
        "CI_high",
        "p",
        "Top25_events",
        "Bottom75_events",
        "Failure_concentration_top25_%"
    ]

    display = summary[
        display_columns
    ].copy()

    display.columns = [
        "Cohort",
        "N",
        "Events",
        "HR",
        "95% CI Low",
        "95% CI High",
        "p",
        "Top 25% Events",
        "Bottom 75% Events",
        "Failure Concentration Top 25%"
    ]

    print(
        display.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.3f}"
        )
    )

    print("\nInterpretation:")

    print(
        "1. Theil–Sen rate defines the robust early "
        "ITH degradation metric."
    )

    print(
        "2. The univariate Cox model is the primary "
        "test of association with later failure hazard."
    )

    print(
        "3. Broad vs Strict tests whether the result "
        "is robust to the 168 h cohort definition."
    )

    print(
        "4. Kaplan–Meier provides visual survival evidence."
    )

    print(
        "5. The top-25% analysis provides intuitive "
        "supporting evidence by showing where failures concentrate."
    )


# =============================================================================
# 15. MAIN
# =============================================================================

if __name__ == "__main__":

    # =========================================================================
    # LOAD
    # =========================================================================

    df = load_data()

    # =========================================================================
    # CALCULATE EARLY RATE + LAST OBSERVATION
    # =========================================================================

    df, ith_columns = add_features(
        df
    )

    # =========================================================================
    # DIAGNOSTIC COHORT
    # =========================================================================

    diagnostic_cohort = diagnose_cohort_steps(
        df
    )

    # =========================================================================
    # BROAD COHORT
    # =========================================================================

    broad = build_cohort(
        df,
        strict=False
    )

    # =========================================================================
    # STRICT COHORT
    # =========================================================================

    strict = build_cohort(
        df,
        strict=True
    )

    # =========================================================================
    # PRINT COHORT SUMMARIES
    # =========================================================================

    print_cohort_summary(
        broad,
        "Broad 168 h"
    )

    print_cohort_summary(
        strict,
        "Strict 168 h"
    )

    # =========================================================================
    # ACTUAL FAILURE TIMES
    # =========================================================================

    print("\n" + "=" * 80)
    print("ACTUAL FAILURE TIMES RETAINED")
    print("=" * 80)

    broad_failures = broad.loc[
        broad["event"].eq(1),
        "duration"
    ].sort_values()

    strict_failures = strict.loc[
        strict["event"].eq(1),
        "duration"
    ].sort_values()

    print(
        "\nBroad failure times:"
    )

    print(
        broad_failures.to_list()
    )

    print(
        "\nStrict failure times:"
    )

    print(
        strict_failures.to_list()
    )

    # =========================================================================
    # FOLLOW-UP CHECK
    # =========================================================================

    print("\n" + "=" * 80)
    print("FOLLOW-UP CHECK")
    print("=" * 80)

    print(
        "\nAll available post-landmark follow-up is retained."
    )

    print(
        "There is no artificial censoring or truncation "
        "at 1000 h or any other fixed follow-up time."
    )

    print(
        f"Broad maximum follow-up: "
        f"{broad['duration'].max():.0f} h"
    )

    print(
        f"Strict maximum follow-up: "
        f"{strict['duration'].max():.0f} h"
    )

    # =========================================================================
    # 1. UNIVARIATE COX
    # =========================================================================

    broad_cox = fit_univariate_cox(
        broad,
        "Broad 168 h"
    )

    strict_cox = fit_univariate_cox(
        strict,
        "Strict 168 h"
    )

    # =========================================================================
    # 2. TOP 25% ANALYSIS
    # =========================================================================

    broad_top25 = top25_analysis(
        broad,
        "Broad 168 h"
    )

    strict_top25 = top25_analysis(
        strict,
        "Strict 168 h"
    )

    # =========================================================================
    # 3. KAPLAN–MEIER
    # =========================================================================

    plot_kaplan_meier(
        broad,
        "Broad 168 h"
    )

    plot_kaplan_meier(
        strict,
        "Strict 168 h"
    )

    # =========================================================================
    # 4. FAILURE-CONCENTRATION PLOT
    # =========================================================================

    plot_failure_concentration(
        broad,
        "Broad 168 h"
    )

    plot_failure_concentration(
        strict,
        "Strict 168 h"
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print_final_summary(
        [
            broad_cox,
            strict_cox
        ],
        [
            broad_top25,
            strict_top25
        ]
    )

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        "\nThe analysis asks whether faster early ITH "
        "degradation is associated with higher subsequent "
        "failure risk."
    )

    print(
        "\nThe early degradation rate is the Theil–Sen slope "
        "calculated from valid ITH measurements <=168 h."
    )

    print(
        "\nThe 168 h landmark defines the post-landmark "
        "risk set."
    )

    print(
        "\nFailures <=168 h are excluded."
    )

    print(
        "\nFailures after 168 h retain their actual failure time."
    )

    print(
        "\nNon-failures retain their actual last observation "
        "as the censoring time."
    )

    print(
        "\nAll available follow-up is retained."
    )
