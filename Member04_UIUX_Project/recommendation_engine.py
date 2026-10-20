
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "datasets" / "member04_recommendation_dataset.csv"

rules = pd.read_csv(CSV_PATH)

severity_values = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

def integrated_recommendation(issue, severity, impact):
    if severity not in severity_values:
        raise ValueError("Severity must be Low, Medium or High")

    if not 1 <= impact <= 5:
        raise ValueError("Impact must be between 1 and 5")

    match = rules[rules["issue_category"] == issue]

    if match.empty:
        return {"status": "No matching recommendation"}

    rule = match.iloc[0]

    priority = round(
        0.6 * (severity_values[severity] / 3)
        + 0.4 * (impact / 5),
        3
    )

    return {
        "issue": issue,
        "severity": severity,
        "impact": impact,
        "priority_score_demo": priority,
        "recommendation": rule["recommended_solution"],
        "explanation": rule["explanation"],
        "guideline": rule["guideline_reference"]
    }

if __name__ == "__main__":
    result = integrated_recommendation(
        issue="low_text_contrast",
        severity="High",
        impact=5
    )

    print("MEMBER 04 RECOMMENDATION ENGINE")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")

    assert result["priority_score_demo"] == 1.0
    print("\nRECOMMENDATION ENGINE TEST PASSED!")
