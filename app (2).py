import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse, ParserError
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from typing import Dict, Any, List

# Configuration
SERPER_API_URL = "https://google.serper.dev/search"
CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = {
        'api_keys': {'serper': '', 'openai': ''},
        'search_type': 'news',
        'results': None
    }

# --------------------------
# Core Functionality
# --------------------------

def serper_news_search(query: str, api_key: str, time_filter: str) -> List[Dict]:
    """Search news with Serper API"""
    time_map = {
        "1 Week": "qdr:w",
        "1 Month": "qdr:m",
        "3 Months": "qdr:3m",
        "6 Months": "qdr:6m",
        "1 Year": "qdr:y"
    }
    
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {
        "q": f"{query} pharmaceutical",
        "tbm": "nws",
        "num": 10,
        "tbs": time_map.get(time_filter, "")
    }
    
    try:
        response = requests.post(SERPER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get('newsResults', [])
    except Exception as e:
        st.error(f"News search failed: {str(e)}")
        return []

def get_clinical_trials(search_term: str) -> pd.DataFrame:
    """Fetch clinical trials data"""
    params = {"query.term": search_term, "pageSize": 100}
    all_studies = []
    
    while True:
        try:
            response = requests.get(CLINICAL_TRIALS_API, params=params)
            data = response.json()
            all_studies.extend(data.get('studies', []))
            if not data.get('nextPageToken'): break
            params['pageToken'] = data['nextPageToken']
        except Exception as e:
            st.error(f"Clinical trials fetch failed: {str(e)}")
            break
    
    # Normalization logic
    normalized = []
    for study in all_studies:
        norm_study = {
            'nctId': study.get('protocolSection', {}).get('identificationModule', {}).get('nctId'),
            'title': study.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle'),
            'status': study.get('protocolSection', {}).get('statusModule', {}).get('overallStatus'),
            'phase': ', '.join(study.get('protocolSection', {}).get('designModule', {}).get('phases', [])),
            'interventions': ', '.join([
                interv.get('name') 
                for interv in study.get('protocolSection', {}).get('armsInterventionsModule', {}).get('interventions', [])
            ])
        }
        normalized.append(norm_study)
    
    return pd.DataFrame(normalized)

def process_news(news_items: List[Dict], openai_key: str) -> Dict:
    """Categorize and summarize news"""
    chat = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_key)
    categories = {"Regulatory": [], "Commercialization": [], "Clinical": []}
    
    for item in news_items:
        response = chat.invoke([
            HumanMessage(f"Categorize:\n{item['title']}\n{item['snippet']}\nOptions: Regulatory, Commercialization, Clinical")
        ])
        category = response.content.strip()
        if category in categories:
            categories[category].append(item)
    
    # Generate summaries
    for cat, items in categories.items():
        if items:
            summary_prompt = f"Summarize these {cat} updates in 3 bullet points: {items}"
            response = chat.invoke([HumanMessage(summary_prompt)])
            categories[cat] = response.content
            
    return categories

# --------------------------
# LangGraph Workflow
# --------------------------

def search_node(state: Dict) -> Dict:
    """Execute search based on type"""
    if state['search_type'] == 'news':
        state['news_raw'] = serper_news_search(
            state['query'],
            state['api_keys']['serper'],
            state['time_filter']
        )
    else:
        state['clinical_raw'] = get_clinical_trials(state['query'])
    return state

def process_node(state: Dict) -> Dict:
    """Process search results"""
    if state['search_type'] == 'news':
        state['news_processed'] = process_news(
            state['news_raw'],
            state['api_keys']['openai']
        )
    else:
        state['clinical_processed'] = state['clinical_raw'].to_dict()
    return state

def report_node(state: Dict) -> Dict:
    """Generate final output"""
    chat = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, openai_api_key=state['api_keys']['openai'])
    
    if state['search_type'] == 'news':
        prompt = f"Generate professional pharma report from: {state['news_processed']}"
    else:
        prompt = f"Analyze clinical trials data: {state['clinical_processed']}"
    
    state['report'] = chat.invoke([HumanMessage(prompt)]).content
    return state

# Build workflow
workflow = StateGraph(dict)
workflow.add_node("search", search_node)
workflow.add_node("process", process_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "process")
workflow.add_edge("process", "report")
workflow.add_edge("report", END)

app = workflow.compile()

# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(
    page_title="Pharma Intelligence Suite",
    layout="wide",
    page_icon="🔬"
)

# Sidebar
with st.sidebar:
    st.title("Configuration")
    st.session_state.state['api_keys']['serper'] = st.text_input(
        "Serper API Key", type="password")
    st.session_state.state['api_keys']['openai'] = st.text_input(
        "OpenAI API Key", type="password")
    
    st.session_state.state['search_type'] = st.radio(
        "Search Type", ["News", "Clinical Trials"])
    
    if st.session_state.state['search_type'] == "News":
        st.session_state.state['time_filter'] = st.selectbox(
            "Time Filter", ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"])

# Main Interface
st.title("Pharma Research Intelligence Platform")

# Search Input
query = st.text_input("Enter search keywords:", key="search_input")
if st.button("Run Analysis"):
    if not all(st.session_state.state['api_keys'].values()):
        st.error("Please provide all API keys")
    else:
        st.session_state.state['query'] = query
        st.session_state.results = app.invoke(st.session_state.state)

# Display Results
if st.session_state.get('results'):
    st.subheader("Analysis Report")
    st.markdown(st.session_state.results['report'])
    
    if st.session_state.state['search_type'] == "News":
        st.download_button(
            label="Download News Data",
            data=pd.DataFrame(st.session_state.results['news_raw']).to_csv(),
            file_name="news_results.csv"
        )
    else:
        st.download_button(
            label="Download Clinical Data",
            data=pd.DataFrame(st.session_state.results['clinical_processed']).to_csv(),
            file_name="clinical_trials.csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    text-align: center;
    padding: 1rem;
    position: relative;
    bottom: 0;
    width: 100%;
}
</style>
<div class="footer">
    <p>Pharma Intelligence Suite • Secure API Handling • Professional Reporting</p>
</div>
""", unsafe_allow_html=True)
