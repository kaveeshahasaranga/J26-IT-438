from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'datasets' / 'member04_recommendation_dataset.csv'
rules = pd.read_csv(CSV_PATH)
severity_values = {'Low': 1, 'Medium': 2, 'High': 3}

# Prototype guidance: proposed implementation advice, not expert-validated outcomes.
EXPLAINABILITY = {
    'low_text_contrast': {
        'user_impact': 'Text may be difficult to read, particularly for users with low vision.',
        'fix_steps': ['Identify text/background pairs with insufficient contrast.', 'Adjust text or background colors.', 'Recheck contrast for normal and large text.'],
        'verification': ['Measure contrast ratios using an accessibility checker.', 'Check text in normal, hover, and focus states.'],
    },
    'small_touch_targets': {
        'user_impact': 'Small interactive controls may be difficult to activate accurately.',
        'fix_steps': ['Identify small buttons, links, and interactive controls.', 'Increase target size or spacing where appropriate.', 'Check controls at mobile viewport sizes.'],
        'verification': ['Measure target dimensions and spacing.', 'Test tapping controls on a mobile device.'],
    },
    'missing_form_labels': {
        'user_impact': 'Users may not know what to enter; assistive technology users may lack accessible field names.',
        'fix_steps': ['Find inputs without visible or accessible labels.', 'Add meaningful labels and associate them with the correct inputs.', 'Check that instructions and required fields are clear.'],
        'verification': ['Inspect accessible names in browser developer tools.', 'Test form fields with a screen reader and keyboard.'],
    },
    'missing_alt_text': {
        'user_impact': 'Users relying on screen readers may miss information conveyed by images.',
        'fix_steps': ['Identify informative and decorative images.', 'Add concise alternative text for informative images.', 'Use empty alt text for purely decorative images when appropriate.'],
        'verification': ['Inspect image alternative text.', 'Navigate the page with a screen reader.'],
    },
    'keyboard_navigation': {
        'user_impact': 'Keyboard-only users may be unable to reach or operate important controls.',
        'fix_steps': ['Identify interactive elements that cannot be reached by keyboard.', 'Use semantic interactive elements and correct focus handling.', 'Ensure focus indicators are visible.'],
        'verification': ['Navigate with Tab, Shift+Tab, Enter, and Space.', 'Check focus order and keyboard operation.'],
    },
    'unclear_error_message': {
        'user_impact': 'Users may struggle to understand and recover from errors.',
        'fix_steps': ['Identify vague or missing error messages.', 'Explain the error in clear language.', 'Provide actionable correction guidance near the relevant field.'],
        'verification': ['Trigger common validation errors.', 'Confirm messages identify the issue and how to resolve it.'],
    },
    'inconsistent_navigation': {
        'user_impact': 'Users may find it harder to locate content across pages.',
        'fix_steps': ['Compare navigation structure across related pages.', 'Use consistent labels, ordering, and placement.', 'Clearly identify the current page where relevant.'],
        'verification': ['Review navigation across multiple pages.', 'Conduct a short task-based usability check.'],
    },
    'missing_page_title': {
        'user_impact': 'Users may have difficulty identifying pages in browser tabs or assistive technology.',
        'fix_steps': ['Check each page for a descriptive HTML title.', 'Add a unique title describing the page purpose.', 'Keep title wording consistent with page content.'],
        'verification': ['Inspect the HTML title element.', 'Confirm browser tabs and screen readers announce meaningful titles.'],
    },
}


def integrated_recommendation(issue, severity, impact):
    if severity not in severity_values:
        raise ValueError('Severity must be Low, Medium or High')
    if isinstance(impact, bool) or not isinstance(impact, int) or not 1 <= impact <= 5:
        raise ValueError('Impact must be an integer between 1 and 5')

    match = rules[rules['issue_category'] == issue]
    if match.empty:
        return {'status': 'No matching recommendation', 'issue': issue}

    rule = match.iloc[0]
    severity_component = round(0.6 * severity_values[severity] / 3, 3)
    impact_component = round(0.4 * impact / 5, 3)
    priority = round(severity_component + impact_component, 3)
    detail = EXPLAINABILITY.get(issue, {})

    return {
        'issue': issue,
        'severity': severity,
        'impact': impact,
        'priority_score_demo': priority,
        'severity_contribution': severity_component,
        'impact_contribution': impact_component,
        'priority_explanation': (
            f'Severity ({severity}) contributes {severity_component:.3f} and '
            f'user impact ({impact}/5) contributes {impact_component:.3f}; '
            f'total priority is {priority:.3f}. Weights are preliminary, not validated.'
        ),
        'recommendation': str(rule['recommended_solution']),
        'explanation': str(rule['explanation']),
        'guideline': str(rule['guideline_reference']),
        'potential_user_impact': detail.get('user_impact', 'Impact requires manual review.'),
        'fix_steps': detail.get('fix_steps', ['Review the issue and apply the recommended solution.']),
        'verification_steps': detail.get('verification', ['Manually verify the improvement.']),
    }


def evaluate_multiple_issues(issue_list):
    results = []
    for item in issue_list:
        result = integrated_recommendation(item['issue'], item['severity'], item['impact'])
        if 'priority_score_demo' in result:
            results.append(result)
    results.sort(key=lambda item: item['priority_score_demo'], reverse=True)
    for index, result in enumerate(results, start=1):
        result['rank'] = index
    return results


if __name__ == '__main__':
    example = integrated_recommendation('missing_form_labels', 'High', 5)
    assert example['priority_score_demo'] == 1.0
    assert example['severity_contribution'] == 0.6
    assert example['impact_contribution'] == 0.4
    cases = [
        {'issue': 'low_text_contrast', 'severity': 'High', 'impact': 5},
        {'issue': 'small_touch_targets', 'severity': 'Medium', 'impact': 4},
        {'issue': 'missing_form_labels', 'severity': 'Low', 'impact': 2},
    ]
    ranked = evaluate_multiple_issues(cases)
    assert [r['priority_score_demo'] for r in ranked] == [1.0, 0.72, 0.36]
    print('ALL ENHANCED RECOMMENDATION ENGINE TESTS PASSED!')
