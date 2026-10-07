# Week 3 — Statistical Analysis and Hypothesis Testing
# Internship project — Pushpak Koppada

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("week3_statistical_analysis_dataset.csv")

# Descriptive statistics
print(df.groupby("Group")["Metric"].describe())

# Welch's independent-samples t-test
control = df.loc[df["Group"] == "Control", "Metric"]
treatment = df.loc[df["Group"] == "Treatment", "Metric"]
t_stat, t_p = stats.ttest_ind(control, treatment, equal_var=False)
print("\nWelch t-test:", t_stat, t_p)

# One-way ANOVA
anova_groups = [df.loc[df["Segment"] == s, "Metric"] for s in ["Low", "Medium", "High"]]
f_stat, anova_p = stats.f_oneway(*anova_groups)
print("One-way ANOVA:", f_stat, anova_p)

# Chi-square test
contingency = pd.crosstab(df["Group"], df["Outcome"])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
print("Chi-square:", chi2, chi_p)

# Visualizations
plt.figure(figsize=(8, 5))
plt.hist(control, bins=18, alpha=0.65, label="Control")
plt.hist(treatment, bins=18, alpha=0.65, label="Treatment")
plt.xlabel("Metric"); plt.ylabel("Frequency")
plt.title("Metric Distribution by Group")
plt.legend(); plt.tight_layout()
plt.savefig("metric_distribution.png", dpi=220)
plt.show()
