import streamlit as st
import requests
import pandas as pd
import json
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configuration
SERPER_API_URL = "https://google.serper.dev/search"
CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_COLUMNS = ['nctId', 'title', 'status', 'phase', 'interventions', 'start_date', 'completion_date']

# --------------------------
# Pydantic State Model
# --------------------------

class PharmaState(BaseModel):
    search_type: str = "news"
    query: Optional[str] = None
    time_filter: str = "1 Week"
    news_raw: List[Dict] = Field(default_factory=list)
    clinical_raw: List[Dict] = Field(default_factory=list)  # Store as list of dicts
    news_processed: Dict[str, Any] = Field(default_factory=dict)
    clinical_processed: Dict[str, Any] = Field(default_factory=dict)
    report: Optional[str] = None

# --------------------------
# Core Functionality
# --------------------------

def sanitize_input(text: str) -> str:
    """Clean user input for API safety"""
    return text.strip('"\'').split('(')[0].strip()

def serper_search(query: str, api_key: str, time_filter: str) -> List[Dict]:
    """Search news using Serper API"""
    time_map = {
        "1 Week": "qdr:w",
        "1 Month": "qdr:m",
        "3 Months": "qdr:3m",
        "6 Months": "qdr:6m",
        "1 Year": "qdr:y"
    }
    
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {
        "q": f"{sanitize_input(query)} pharmaceutical",
        "tbm": "nws",
        "num": 10,
        "tbs": time_map.get(time_filter, "")
    }
    
    try:
        response = requests.post(SERPER_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json().get('newsResults', [])
    except Exception as e:
        st.error(f"News search error: {str(e)}")
        return []

def fetch_clinical_trials(term: str) -> List[Dict]:
    """Fetch clinical trials data and return as list of dicts"""
    params = {"query.term": sanitize_input(term), "pageSize": 100}
    all_studies = []
    
    try:
        while True:
            response = requests.get(CLINICAL_TRIALS_API, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            all_studies.extend(data.get('studies', []))
            
            page_token = data.get('nextPageToken')
            if not page_token: break
            params['pageToken'] = page_token
    except Exception as e:
        st.error(f"Clinical trials error: {str(e)}")
        return []
    
    normalized = []
    for study in all_studies:
        protocol = study.get('protocolSection', {})
        normalized.append({
            'nctId': protocol.get('identificationModule', {}).get('nctId'),
            'title': protocol.get('identificationModule', {}).get('briefTitle'),
            'status': protocol.get('statusModule', {}).get('overallStatus'),
            'phase': ', '.join(protocol.get('designModule', {}).get('phases', [])),
            'interventions': ', '.join([
                i.get('name') for i in 
                protocol.get('armsInterventionsModule', {}).get('interventions', [])
            ]),
            'start_date': protocol.get('statusModule', {}).get('startDateStruct', {}).get('date'),
            'completion_date': protocol.get('statusModule', {}).get('completionDateStruct', {}).get('date')
        })
    
    return normalized

def analyze_news(items: List[Dict], api_key: str) -> Dict:
    """Categorize and summarize news using AI"""
    if not items: return {}
    
    try:
        chat = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=api_key)
        categories = {"Regulatory": [], "Commercialization": [], "Clinical": []}
        
        for item in items:
            response = chat.invoke([HumanMessage(
                f"Categorize:\n{item['title']}\n{item['snippet']}\nOptions: {', '.join(categories.keys())}"
            )])
            category = response.content.strip()
            if category in categories:
                categories[category].append(item)
        
        for cat, items in categories.items():
            if items:
                summary = chat.invoke([HumanMessage(
                    f"Summarize {cat} updates in 3 bullet points: {items}"
                )])
                categories[cat] = summary.content
                
        return categories
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return {}

# --------------------------
# LangGraph Workflow
# --------------------------

def search_step(state: PharmaState) -> PharmaState:
    """Execute search based on type"""
    if not state.query or not state.query.strip():
        st.error("Please enter valid search terms")
        return state
    
    try:
        if state.search_type == "news":
            state.news_raw = serper_search(
                state.query,
                st.secrets["SERPER_API_KEY"],
                state.time_filter
            )
        else:
            state.clinical_raw = fetch_clinical_trials(state.query)
        return state
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return state

def process_step(state: PharmaState) -> PharmaState:
    """Process search results"""
    try:
        if state.search_type == "news":
            state.news_processed = analyze_news(
                state.news_raw,
                st.secrets["OPENAI_API_KEY"]
            )
        else:
            df = pd.DataFrame(state.clinical_raw)
            state.clinical_processed = {
                "total": len(df),
                "phases": df['phase'].value_counts().to_dict(),
                "statuses": df['status'].value_counts().to_dict()
            }
        return state
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return state

def report_step(state: PharmaState) -> PharmaState:
    """Generate final report"""
    try:
        chat = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, 
                         openai_api_key=st.secrets["OPENAI_API_KEY"])
        
        if state.search_type == "news":
            prompt = f"Generate pharma industry report with markdown:\n{state.news_processed}"
        else:
            prompt = f"Analyze clinical trials data:\n{state.clinical_processed}"
        
        state.report = chat.invoke([HumanMessage(prompt)]).content
        return state
    except Exception as e:
        st.error(f"Reporting error: {str(e)}")
        return state

# Build workflow
workflow = StateGraph(PharmaState)
workflow.add_node("search", search_step)
workflow.add_node("process", process_step)
workflow.add_node("report", report_step)

workflow.set_entry_point("search")
workflow.add_edge("search", "process")
workflow.add_edge("process", "report")
workflow.add_edge("report", END)

app = workflow.compile()

# --------------------------
# Streamlit Interface
# --------------------------

st.set_page_config(
    page_title="Pharma Intelligence Suite",
    layout="wide",
    page_icon="🔬"
)

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = PharmaState().model_dump()

# Sidebar Configuration
with st.sidebar:
    st.title("Configuration")
    search_type = st.radio(
        "Search Type", ["News", "Clinical Trials"],
        index=0 if st.session_state.state['search_type'] == "news" else 1
    )
    time_filter = st.selectbox(
        "Time Filter", ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
        index=1
    ) if search_type == "News" else "1 Week"

# Main Interface
st.title("Pharma Research Platform")

# Search Input
query = st.text_input("Enter search keywords:", key="search_input")
if st.button("Run Analysis"):
    if not query.strip():
        st.error("Please enter search terms")
        st.stop()
    
    # Prepare fresh state
    new_state = PharmaState(
        search_type=search_type.lower(),
        query=query.strip(),
        time_filter=time_filter,
        news_raw=[],
        clinical_raw=[],
        news_processed={},
        clinical_processed={},
        report=None
    )
    
    try:
        # Execute workflow
        for step in app.stream(new_state):
            node_name, node_state = next(iter(step.items()))
            st.session_state.state = node_state.model_dump()
        
        st.success("Analysis completed successfully!")
    except ValidationError as e:
        st.error(f"Validation error: {e}")
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# Display Results
if st.session_state.state.get('report'):
    st.subheader("Analysis Report")
    st.markdown(st.session_state.state['report'])
    
    # Download Options
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.state['search_type'] == "news":
            st.download_button(
                "Download News Data",
                data=pd.DataFrame(st.session_state.state['news_raw']).to_csv(),
                file_name="news_results.csv"
            )
    with col2:
        if st.session_state.state['search_type'] == "clinical_trials":
            st.download_button(
                "Download Clinical Data",
                data=pd.DataFrame(st.session_state.state['clinical_raw']).to_csv(),
                file_name="clinical_trials.csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    text-align: center;
    padding: 1rem;
    background-color: #f0f2f6;
    border-radius: 0.5rem;
    margin-top: 2rem;
}
</style>
<div class="footer">
    <p>Pharma Intelligence Suite | Secure Processing | Professional Reporting</p>
</div>
""", unsafe_allow_html=True)
