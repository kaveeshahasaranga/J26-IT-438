from pathlib import Path
import pandas as pd
import streamlit as st
from recommendation_engine import integrated_recommendation, evaluate_multiple_issues, rules

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'datasets' / 'member04_ui_quality_dataset.csv'
SCREENSHOT_DIR = BASE_DIR / 'screenshots'

st.set_page_config(page_title='Member 04 - UI/UX Decision Engine', page_icon='📊', layout='wide')
st.title('Intelligent UI/UX Evaluation & Decision Engine')
st.caption('Member 04 | PP1 Research Prototype | Rule-Based Decision Support')

if not DATA_PATH.is_file():
    st.error('UI quality dataset file not found.')
    st.stop()

df = pd.read_csv(DATA_PATH, dtype=str).fillna('')
issue_categories = rules['issue_category'].dropna().unique().tolist()
if not issue_categories:
    st.error('No recommendation rules available.')
    st.stop()


def show_explanation(result):
    st.metric('Rule-Based Priority Score', f"{result['priority_score_demo']:.3f}")
    st.write('**Why this priority?**')
    st.info(result['priority_explanation'])
    st.write('**Score contributions**')
    st.progress(min(1.0, result['priority_score_demo']))
    c1, c2 = st.columns(2)
    c1.metric('Severity contribution (max 0.60)', f"{result['severity_contribution']:.3f}")
    c2.metric('Impact contribution (max 0.40)', f"{result['impact_contribution']:.3f}")
    st.write('**Recommended Solution**')
    st.success(result['recommendation'])
    st.write('**Recommendation Explanation**')
    st.info(result['explanation'])
    st.write('**Potential User Impact**')
    st.write(result['potential_user_impact'])
    st.write('**Step-by-Step Fix Instructions**')
    for i, step in enumerate(result['fix_steps'], 1):
        st.write(f'{i}. {step}')
    st.write('**How to Verify the Fix**')
    for i, step in enumerate(result['verification_steps'], 1):
        st.write(f'{i}. {step}')
    st.write('**Guideline Reference**')
    st.write(result['guideline'])


st.header('1. Dataset Overview')
pending_count = (df['expert_quality_rating'].str.strip() == '').sum()
duplicates = df.duplicated(subset=['page_url', 'device_type']).sum()
c1, c2, c3 = st.columns(3)
c1.metric('Interfaces', len(df))
c2.metric('Expert Ratings Pending', int(pending_count))
c3.metric('Duplicate URL/Device Pairs', int(duplicates))
st.dataframe(df[['interface_id', 'page_url', 'page_type', 'device_type', 'label_status']], use_container_width=True, hide_index=True)

st.header('2. Interface Screenshot Preview')
if not df.empty:
    selected_id = st.selectbox('Select Interface', df['interface_id'].tolist())
    selected_row = df[df['interface_id'] == selected_id].iloc[0]
    st.write('Website:', selected_row['page_url'])
    st.write('Device:', selected_row['device_type'])
    image_path = SCREENSHOT_DIR / selected_row['screenshot_filename']
    if image_path.is_file():
        st.image(str(image_path), width=800)
    else:
        st.warning('Screenshot not found for this interface.')
st.caption('Screenshot preview does not automatically detect or verify UI/UX issues.')

st.header('3. Single Issue Evaluation')
st.caption('Evaluate one manually entered UI/UX issue.')
issue = st.selectbox('Issue Category', issue_categories, key='single_issue')
severity = st.selectbox('Severity', ['Low', 'Medium', 'High'], key='single_severity')
impact = st.slider('User Impact', 1, 5, 3, key='single_impact')
if st.button('Evaluate Issue', type='primary'):
    st.session_state.single_result = integrated_recommendation(issue, severity, impact)
    st.session_state.single_inputs = (issue, severity, impact)
if 'single_result' in st.session_state:
    if st.session_state.get('single_inputs') != (issue, severity, impact):
        st.caption('Inputs changed. Click Evaluate Issue to update this result.')
    result = st.session_state.single_result
    if 'priority_score_demo' in result:
        show_explanation(result)
    else:
        st.warning(result.get('status', 'No recommendation found'))

st.divider()
st.header('4. Multi-Issue Prioritization')
st.caption('Add multiple manually assessed issues, rank them, and review actionable explanations.')
if 'issue_list' not in st.session_state:
    st.session_state.issue_list = []
if 'ranked_results' not in st.session_state:
    st.session_state.ranked_results = []
multi_issue = st.selectbox('Select Issue', issue_categories, key='multi_issue')