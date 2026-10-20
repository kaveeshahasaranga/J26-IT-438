
from recommendation_engine import (
    integrated_recommendation,
    rules
)

print("=" * 45)
print("MEMBER 04 UI/UX DECISION ENGINE")
print("=" * 45)

print("\nAvailable UI/UX Issues:")
for issue in rules["issue_category"]:
    print("-", issue)

print("\nEnter issue information")

issue = input("Issue category: ").strip()
severity = input(
    "Severity (Low/Medium/High): "
).strip().capitalize()

try:
    impact = int(input("User impact (1-5): "))

    result = integrated_recommendation(
        issue=issue,
        severity=severity,
        impact=impact
    )

    print("\n--- EVALUATION RESULT ---")
    for key, value in result.items():
        print(f"{key}: {value}")

except ValueError as error:
    print("Input error:", error)
