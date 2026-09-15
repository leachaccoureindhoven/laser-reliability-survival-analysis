## Early ITH Degradation — 168 h Landmark Survival Validation

### Purpose

This analysis evaluates whether **faster early degradation of laser threshold current (ITH)** is associated with a higher risk of subsequent device failure.

The analysis uses a **168 h landmark approach**: early degradation is quantified using measurements collected during the first 168 h, and only devices that survive and remain observable through the 168 h landmark are included in the subsequent survival analysis.

The analysis is **associational rather than predictive**. The objective is to determine whether early ITH degradation is a meaningful indicator of later reliability risk.

---

### Analysis workflow

The script performs the following steps:

1. **Automatically identifies all available ITH timepoints**

   * Columns following the `ITH_<time>` naming convention are detected automatically.
   * This avoids hard-coding a fixed set of measurement times.

2. **Calculates the early ITH degradation rate**

   * All valid ITH measurements collected at `t ≤ 168 h` are used.
   * At least **3 valid early measurements** are required.
   * The degradation rate is estimated using a **Theil–Sen slope**, providing a robust estimate that is less sensitive to individual noisy or anomalous measurements than ordinary least-squares regression.

3. **Defines the 168 h landmark cohorts**

   Two cohorts are analysed:

   **Broad cohort**

   * Valid `ITH_0`
   * At least 3 valid ITH measurements during the first 168 h
   * No failure at or before 168 h
   * Follow-up reaching at least 168 h

   **Strict cohort**

   * All Broad cohort criteria
   * Additionally requires an actual `ITH_168` measurement

   The Broad cohort maximizes the available sample size, while the Strict cohort provides a sensitivity analysis based on devices with an explicit 168 h measurement.

4. **Defines post-landmark survival**

   * Failures at or before 168 h are excluded from the landmark analysis.
   * Failures after 168 h are retained as events at their **actual failure time**.
   * Devices that do not fail are censored at their **actual last valid observation**.
   * Early degradation is defined using measurements up to 168 h.
   *After the 168 h landmark, each device is followed until failure or its last available observation.


5. **Tests the association using a univariate Cox model**

   * The primary statistical analysis is a **univariate Cox proportional-hazards model** with early ITH degradation rate as the explanatory variable.
   * The early degradation rate is standardized to one standard deviation.
   * Therefore, the reported hazard ratio (HR) represents the change in subsequent failure hazard associated with a **1-SD increase in early degradation rate**. (For every increase of 1 standard deviation in the early degradation rate, the model estimates how much the risk of failing later changes).

6. **Compares survival using Kaplan–Meier analysis**

   * Devices are divided into:

     * **Top 25%:** fastest early degraders
     * **Bottom 75%:** slower early degraders
   * Kaplan–Meier curves are used to visualize differences in post-168 h survival.

7. **Quantifies failure concentration**

   * The analysis reports what fraction of all post-landmark failures occurs within the fastest-degrading 25% of devices.
   * This provides an intuitive complementary measure of whether failures are concentrated among devices showing faster early degradation.

---

### Statistical outputs

For each cohort, the script reports:

* Number of devices
* Number of post-landmark failures
* Number of censored devices
* Median follow-up time
* Maximum follow-up time
* Cox hazard ratio
* 95% confidence interval
* Cox model p-value
* Number of failures in the fastest 25%
* Number of failures in the remaining 75%
* Failure rate in each group
* Fraction of all failures occurring in the fastest 25%

The script also generates:

* Kaplan–Meier survival plots
* Failure-concentration plots

---

### Interpretation

The **primary evidence** comes from the univariate Cox model.

A hazard ratio greater than 1 indicates that devices with a higher early ITH degradation rate have a higher subsequent failure hazard. Statistical significance is assessed using the Cox model p-value and confidence interval.

The Broad and Strict cohorts provide a robustness check against the exact definition of the 168 h landmark.

The Kaplan–Meier and top-25% failure-concentration analyses are used as **supporting evidence**, helping visualize and interpret the relationship between early degradation and later failure.

Importantly, the analysis does **not** attempt to predict individual device failure times. It evaluates whether the magnitude of early ITH degradation is associated with subsequent reliability.

---

### Landmark definition

```text
Early period
0 ─────────────────────────────── 168 h
│                                 │
│  Calculate Theil–Sen slope      │ Landmark
│  from valid ITH measurements    │
│                                 │
└─────────────────────────────────┘
                                  │
                                  ▼
                         Post-landmark risk set
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                 Failure                   No failure
                    │                           │
             Actual failure time        Last observation
                    │                           │
                    ▼                           ▼
                  Event                    Censored
```

Thus, the analysis separates **early degradation measurement** from **subsequent failure observation**, avoiding the use of future information when defining the early degradation metric.

---

### Key methodological choices

| Component                    | Definition                          |
| ---------------------------- | ----------------------------------- |
| Landmark                     | 168 h                               |
| Early degradation metric     | Theil–Sen slope of ITH vs. time     |
| Minimum early measurements   | 3                                   |
| Early failures               | Excluded                            |
| Primary model                | Univariate Cox proportional hazards |
| Cox predictor                | Early ITH degradation rate          |
| HR scaling                   | Per 1-SD increase                   |
| Survival visualization       | Kaplan–Meier                        |
| Group comparison             | Fastest 25% vs. remaining 75%       |
| Event time                   | Actual failure milestone            |
| Censoring time               | Actual last observation             |
| Follow-up cutoff             | None                                |
| Artificial 1000 h truncation | None                                |

---

### Files

The main analysis script is:

```text
early_ith_degradation_landmark_survival.py
```

It requires a CSV dataset containing `ITH_<time>` measurements and the corresponding failure information, including:

```text
ITH_0
ITH_8
ITH_24
...
FAIL_ITH0
FAIL_ITH0_milestone
```

The script automatically discovers available `ITH_<time>` columns, so additional measurement timepoints can be included without modifying the timepoint list manually.

### Main dependencies

```text
numpy
pandas
scipy
lifelines
matplotlib
```

### Summary

This analysis provides a focused validation of the hypothesis:

> **Devices exhibiting faster ITH degradation during the first 168 h are more likely to experience subsequent failure.**

The hypothesis is evaluated using a robust early-degradation metric, a 168 h landmark survival framework, univariate Cox regression, Kaplan–Meier survival analysis, and failure-concentration analysis, while retaining the complete available post-landmark follow-up.




