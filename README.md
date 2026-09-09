# laser-reliability-survival-analysis
## How to Use

This repository contains Python code for performing a **168-hour landmark survival analysis of laser reliability** using InP Fabry–Pérot laser data.

The analysis uses early degradation of the threshold current (`ITH`) to investigate its association with **subsequent laser failure**.

### 1. Prepare the input dataset

The analysis requires a CSV file containing the laser-level measurements and reliability information.

The input dataset should contain, at minimum, the following columns:

* `FAIL_ITH0` — failure indicator
* `FAIL_ITH0_milestone` — failure time in hours
* `ITH_0` — threshold current at 0 h
* `ITH_<time>` — threshold-current measurements at different timepoints, for example:

  * `ITH_0`
  * `ITH_8`
  * `ITH_24`
  * `ITH_48`
  * `ITH_96`
  * `ITH_144`
  * `ITH_168`
* `Wafer number` — wafer identifier
* `cavity length` — laser cavity length
* `distance_to_edge` — distance of the device from the wafer edge
* `Climate_Chamber_Ambient_Temperature [degC]` — test temperature
* `Stress_Current_Density [kA/cm2]` — applied stress current density

The script automatically detects all columns following the format:

```text
ITH_<time>
```

where `<time>` is the measurement time in hours.

For example:

```text
ITH_0
ITH_8
ITH_24
ITH_48
ITH_96
ITH_144
ITH_168
```

Additional ITH measurements at later times can also be present.

---

### 2. Set the input file path

Open the Python script and modify:

```python
INPUT_FILE = r"C:\path\to\your\data.csv"
```

For example:

```python
INPUT_FILE = (
    r"C:\Users\username\Downloads"
    r"\laser_reliability_dataset.csv"
)
```

No other changes to the dataset are required if the column names match those expected by the script.

---

## 3. Run the analysis

The script can be executed using Python, Spyder, Jupyter-compatible environments, or another Python IDE.

For example, from a terminal:

```bash
python landmark_survival_analysis.py
```

The script will then:

1. Load the dataset.
2. Automatically identify the available ITH timepoints.
3. Calculate the early ITH degradation rate.
4. Construct the 168-hour landmark cohorts.
5. Perform Cox proportional-hazards analyses.
6. Perform wafer-level diagnostics.
7. Generate the analysis figures.
8. Print the statistical results to the console.

---

## 4. 168-hour landmark analysis

The analysis uses **168 h as the landmark time**.

The purpose is to separate information available during early life from failures occurring later.

Only devices that are still eligible at the 168-hour landmark are included in the survival analysis.

### Early ITH degradation rate

For each laser, the script uses all available ITH measurements from:

```text
0 h ≤ time ≤ 168 h
```

A minimum of **three ITH measurements** is required.

The early degradation rate is calculated using a **Theil–Sen slope**, which provides a robust estimate of the change in ITH with time.

Importantly, measurements after 168 h are **not used to calculate the early degradation rate**.

---

## 5. Two analysis cohorts

Two cohorts are constructed to assess the robustness of the analysis.

### Broad cohort

A laser is included in the Broad cohort if it has:

* a valid `ITH_0`;
* at least 3 ITH measurements between 0 and 168 h;
* no failure at or before 168 h;
* observations extending to at least 168 h.

An exact `ITH_168` measurement is **not required**.

Therefore, a laser can enter the Broad cohort if its last available measurement is at or after 168 h, even if the measurement is not exactly at 168 h.

### Strict cohort

The Strict cohort applies the same criteria as the Broad cohort, but additionally requires:

```text
ITH_168
```

to be available.

Thus, the Strict cohort provides a more restrictive sensitivity analysis based on an explicitly observed 168-hour measurement.

---

## 6. Treatment of failures

The landmark analysis distinguishes failures occurring before and after the 168-hour landmark.

### Failures at or before 168 h

Failures satisfying:

```text
failure time ≤ 168 h
```

are excluded from the landmark survival cohort.

This prevents information from after the failure from being used to define a post-landmark survival analysis.

### Failures after 168 h

Failures satisfying:

```text
failure time > 168 h
```

are treated as events in the landmark analysis.

Therefore, a failure exactly at **168 h is not considered a post-landmark event**.

---

## 7. Survival time

The landmark time is:

```text
168 h
```

For a device that subsequently fails:

```text
duration = failure time
event = 1
```

For a device that does not fail during the available observation period:

```text
duration = last available observation time
event = 0
```

Conceptually, the survival analysis starts at the 168-hour landmark and evaluates the risk of **subsequent failure**.

---

## 8. Cox proportional-hazards model

The main analysis uses a six-variable Cox proportional-hazards model:

1. Stress current density
2. Early ITH degradation rate
3. Cavity length
4. Temperature
5. Distance to wafer edge
6. `ITH_0`

The continuous predictors are standardized before fitting the Cox model.

The model reports:

* Hazard ratio (HR)
* 95% confidence interval
* p-value
* Concordance index (C-index)
* Partial log-likelihood

An HR greater than 1 indicates an increased hazard of subsequent failure per one standard deviation increase in the corresponding standardized predictor.

An HR below 1 indicates a lower hazard.

---

## 9. EPV and interpretation

Because the number of failures is limited, the script calculates the **events-per-variable (EPV)**:

```text
EPV = number of events / number of model variables
```

A reference threshold of:

```text
EPV = 5
```

is used as a warning threshold.

Importantly, the script **does not automatically remove variables when EPV is low**.

Instead, it:

* fits the complete six-variable Model 4;
* clearly flags the model as exploratory when EPV < 5;
* additionally fits a reduced stress-current-density-only model as a sensitivity analysis.

This keeps the complete analysis transparent while indicating the limitations associated with the small number of observed failures.

---

## 10. Wafer-level diagnostics

The analysis also evaluates whether there are indications of wafer-to-wafer differences in failure behavior.

The script performs:

* an unadjusted wafer-versus-failure chi-square test;
* a Cox likelihood-ratio test comparing models with and without wafer terms;
* Kruskal–Wallis testing of Cox martingale residuals across wafers.

These analyses are treated as **diagnostic/exploratory**, particularly when the number of failures is small.

A statistically significant wafer-related diagnostic should therefore not automatically be interpreted as proof of an independent wafer effect.

---

## 11. Generated figures

The script generates several diagnostic figures.

### Figure 1 — Early ITH degradation rate

Comparison of the early Theil–Sen ITH degradation rate between:

* censored devices;
* devices that subsequently failed.

Results are shown for both Broad and Strict cohorts.

### Figure 2 — Cavity length

Comparison of cavity length between censored and subsequently failed devices.

### Figure 3 — Stress current density

Comparison of stress current density between censored and subsequently failed devices.

### Figure 4 — Early ITH degradation rate by outcome

A dedicated comparison of early ITH degradation rate between the two outcome groups for both cohorts.

### Figure 5 — Martingale residuals by wafer

Cox Model 4 martingale residuals are shown by wafer for the Broad and Strict cohorts.

---

## 12. Required Python packages

The analysis requires:

```text
numpy
pandas
matplotlib
scipy
lifelines
```

They can be installed using:

```bash
pip install numpy pandas matplotlib scipy lifelines
```

---

## 13. Important interpretation

This analysis is designed to evaluate whether **early ITH degradation measured during the first 168 h is associated with subsequent laser failure**.

The early degradation rate is calculated exclusively from measurements available up to 168 h, while events are restricted to failures occurring **strictly after 168 h**.

Therefore, the analysis should be interpreted as an assessment of an **early-life reliability indicator and its association with later failure**, rather than as definitive proof of an independently predictive failure model.

The Broad and Strict cohorts provide complementary analyses:

* **Broad:** maximizes the number of usable devices while requiring observation through the 168-hour landmark.
* **Strict:** requires an exact 168-hour ITH measurement and therefore provides a more restrictive sensitivity analysis.

The results should be interpreted in light of the limited number of observed failures and the resulting uncertainty in multivariable Cox estimates.

