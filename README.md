# Employee_Attriction_Analysis-using-python
Python-based HR analytics project predicting employee attrition from IBM dataset (1470 employees) using Random Forest - 82% accuracy.
overview = """
A Business Intelligence project that analyzes HR data to identify
employee attrition trends, key risk factors, and retention strategies.

The project analyzes 1,470 employee records to understand:
- Attrition rate and employee risk
- Department-wise and role-wise attrition trends
- Factors like salary, overtime, and job satisfaction
- Employee risk prediction and retention recommendations

BI Framework:
Facts -> Trends -> Drivers -> Risks -> Recommended Actions
"""

if __name__ == "__main__":
    print(project_title)
    print(overview)
dataset = {
    "name": "IBM HR Analytics Employee Attrition & Performance",
    "employees": 1470,
    "attributes": 35,
    "target": "Attrition (Yes/No)"
}

tech_stack = [
    "Python", "Pandas", "NumPy", "Scikit-learn",
    "Matplotlib", "Seaborn", "Jupyter Notebook"
]

features = [
    "Data Cleaning and HR KPIs",
    "Department-wise and Role-wise Attrition Analysis",
    "Salary, Overtime and Satisfaction Analysis",
    "Random Forest Prediction Model",
    "Employee Risk Prediction",
    "Business Insights and Retention Strategies"
]

model = {
    "algorithm": "Random Forest Classifier",
    "test_split": "20%",
    "accuracy": "~82%",
    "roc_auc": "~0.78",
    "top_factors": [
        "Monthly Income", "Age", "Total Working Years",
        "Years at Company", "OverTime"
    ]
}

key_findings = [
    "Overall attrition rate: 16.1%",
    "Sales department attrition: 20.6%",
    "Sales Representative attrition: 39.8%",
    "Overtime attrition: 30.5% vs. 10.4%",
    "Median income: $3,202 (left) vs. $5,204 (stayed)",
    "Attrition decreases as job satisfaction increases"
]

if __name__ == "__main__":
    print(project_title)
    print("\nDataset:", dataset)
    print("\nTech Stack:", ", ".join(tech_stack))

    print("\nProject Features:")
    for feature in features:
        print("-", feature)

    print("\nModel Performance:", model)

    print("\nKey Findings:")
    for finding in key_findings:

    
project_files = {
    "notebook": "Employee_attrition_analysis_ram(1).ipynb",
    "dependencies": "requirements.txt",
    "dataset": "data/HR-Employee-Attrition.csv",
    "charts": "assets/",
    "report": "Employee_attriction_analysis_ram.docx",
    "script": "generate_assets.py"
}

file_purposes = {
    "notebook": "Data cleaning, KPIs, EDA, ML model and risk prediction",
    "dependencies": "Required Python libraries",
    "dataset": "Local dataset backup",
    "charts": "Project report visualizations",
    "report": "Detailed project documentation",
    "script": "Regenerates charts and metrics"
}

for file, name in project_files.items():
    print(f"{name} - {file_purposes[file]}")

        print("-", finding)

