"""
Generates KPI metrics, model scores, and chart images for HR attrition report.
Execute this script once to create /assets folder data and metrics.json file.
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

CSV_FILE = "data/HR-Employee-Attrition.csv"
OUTPUT_FOLDER = "assets"

# ---------------------------------------------------------------------------
# 1. DATA LOADING AND PRE-PROCESSING
# ---------------------------------------------------------------------------
data = pd.read_csv(CSV_FILE)
data.columns = [col.strip() for col in data.columns]

# Remove columns where all values are same (no information)
cols_constant = [col for col in data.columns if data[col].nunique() == 1]
data = data.drop(columns=cols_constant)

# Remove nulls if any
data = data.dropna()

# ---------------------------------------------------------------------------
# 2. KEY PERFORMANCE INDICATORS
# ---------------------------------------------------------------------------
total_emp = len(data)
attr_rate = (data["Attrition"] == "Yes").mean()
avg_salary = data["MonthlyIncome"].mean()
avg_exp = data["YearsAtCompany"].mean()
avg_job_sat = data["JobSatisfaction"].mean()
risky_emp_ratio = (
    (data["OverTime"] == "Yes") & (data["JobSatisfaction"] <= 2)
).mean()

kpi_data = {
    "total_employees": int(total_emp),
    "attrition_rate_pct": round(attr_rate * 100, 1),
    "avg_monthly_income": round(avg_salary, 0),
    "avg_years_at_company": round(avg_exp, 1),
    "avg_job_satisfaction_out_of_4": round(avg_job_sat, 2),
    "high_risk_employee_share_pct": round(risky_emp_ratio * 100, 1),
}

# ---------------------------------------------------------------------------
# 3. KEY DRIVERS ANALYSIS
# ---------------------------------------------------------------------------
dept_wise = (
    data.groupby("Department")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).sort_values(ascending=False)
)
role_wise = (
    data.groupby("JobRole")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).sort_values(ascending=False)
)
ot_wise = (
    data.groupby("OverTime")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100)
)
income_stats = data.groupby("Attrition")["MonthlyIncome"].median()
sat_wise = (
    data.groupby("JobSatisfaction")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100)
)

driver_data = {
    "top_department_by_attrition": dept_wise.index[0],
    "top_department_attrition_pct": round(dept_wise.iloc[0], 1),
    "lowest_department_by_attrition": dept_wise.index[-1],
    "lowest_department_attrition_pct": round(dept_wise.iloc[-1], 1),
    "top_role_by_attrition": role_wise.index[0],
    "top_role_attrition_pct": round(role_wise.iloc[0], 1),
    "overtime_attrition_pct": round(ot_wise.get("Yes", 0), 1),
    "no_overtime_attrition_pct": round(ot_wise.get("No", 0), 1),
    "median_income_stayed": round(income_stats.get("No", 0), 0),
    "median_income_left": round(income_stats.get("Yes", 0), 0),
    "lowest_satisfaction_attrition_pct": round(sat_wise.iloc[0], 1),
    "highest_satisfaction_attrition_pct": round(sat_wise.iloc[-1], 1),
}

# ---------------------------------------------------------------------------
# 4. MACHINE LEARNING MODEL - Random Forest for attrition prediction
# ---------------------------------------------------------------------------
model_data = data.copy()
y_target = (model_data["Attrition"] == "Yes").astype(int)

# EmployeeNumber is an ID column, not useful for prediction
model_data = model_data.drop(columns=["Attrition", "EmployeeNumber"])

cat_features = model_data.select_dtypes(exclude=[np.number]).columns.tolist()
for col in cat_features:
    le = LabelEncoder()
    model_data[col] = le.fit_transform(model_data[col])

X_train, X_test, y_train, y_test = train_test_split(
    model_data, y_target, test_size=0.2, random_state=42, stratify=y_target
)

rf_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy_pct": round(accuracy_score(y_test, y_pred) * 100, 1),
    "roc_auc": round(roc_auc_score(y_test, y_prob), 3),
    "n_train": len(X_train),
    "n_test": len(X_test),
}

importance = pd.Series(rf_model.feature_importances_, index=model_data.columns).sort_values(ascending=False)
top_8_features = importance.head(8)

# ---------------------------------------------------------------------------
# 5. VISUALIZATIONS
# ---------------------------------------------------------------------------
# Chart 1 - Department wise attrition
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=dept_wise.values, y=dept_wise.index, ax=ax, color="#4C72B0")
ax.set_xlabel("Attrition Rate (%)")
ax.set_title("Attrition Rate by Department")
for i, v in enumerate(dept_wise.values):
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/chart1_dept_attrition.png")
plt.close()

# Chart 2 - Job role wise
top_6_roles = role_wise.head(6)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=top_6_roles.values, y=top_6_roles.index, ax=ax, color="#DD8452")
ax.set_xlabel("Attrition Rate (%)")
ax.set_title("Highest-Risk Job Roles")
for i, v in enumerate(top_6_roles.values):
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/chart2_role_attrition.png")
plt.close()

# Chart 3 - Income comparison
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.boxplot(data=data, x="Attrition", y="MonthlyIncome", hue="Attrition", ax=ax, palette=["#55A868", "#C44E52"], legend=False)
ax.set_title("Monthly Income: Stayed vs Left")
ax.set_ylabel("Monthly Income (USD)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/chart3_income_attrition.png")
plt.close()

# Chart 4 - Satisfaction vs attrition
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.barplot(x=sat_wise.index, y=sat_wise.values, ax=ax, color="#8172B2")
ax.set_xlabel("Job Satisfaction (1=Low, 4=High)")
ax.set_ylabel("Attrition Rate (%)")
ax.set_title("Attrition Rate by Job Satisfaction")
for i, v in enumerate(sat_wise.values):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/chart4_satisfaction_attrition.png")
plt.close()

# Chart 5 - Feature importance
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(x=top_8_features.values, y=top_8_features.index, ax=ax, color="#4C72B0")
ax.set_xlabel("Relative Importance")
ax.set_title("Top Drivers of Attrition (Model Feature Importance)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/chart5_feature_importance.png")
plt.close()

# ---------------------------------------------------------------------------
# 6. SAVE FINAL JSON
# ---------------------------------------------------------------------------
final_output = {
    "kpis": kpi_data,
    "drivers": driver_data,
    "model_metrics": metrics,
    "top_features": {k: round(v, 3) for k, v in top_8_features.items()},
}
with open("metrics.json", "w") as f:
    json.dump(final_output, f, indent=2)

print(json.dumps(final_output, indent=2))
print("\nAll charts saved to /assets. metrics.json written.")
