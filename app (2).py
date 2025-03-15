import streamlit as st
import requests
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List, Optional

# Configuration
SERPER_API_URL = "https://google.serper.dev/search"
CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

# --------------------------
# Pydantic State Model
# --------------------------

class PharmaResearchState(BaseModel):
    api_keys: Dict[str, str] = Field(default_factory=dict)
    search_type: str = "news"
    query: Optional[str] = None
    time_filter: str = "1 Week"
    news_raw: List[Dict] = Field(default_factory=list)
    clinical_raw: pd.DataFrame = Field(default_factory=pd.DataFrame)
    news_processed: Dict[str, Any] = Field(default_factory=dict)
    clinical_processed: Dict[str, Any] = Field(default_factory=dict)
    report: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

# --------------------------
# Core Functionality
# --------------------------

def sanitize_input(text: str) -> str:
    """Sanitize user input for API queries"""
    return text.strip('"\'').split('(')[0].strip()

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
    """Fetch and normalize clinical trials data"""
    params = {"query.term": sanitize_input(search_term), "pageSize": 100}
    all_studies = []
    
    try:
        while True:
            response = requests.get(CLINICAL_TRIALS_API, params=params)
            response.raise_for_status()
            data = response.json()
            all_studies.extend(data.get('studies', []))
            if not data.get('nextPageToken'):
                break
            params['pageToken'] = data['nextPageToken']
    except Exception as e:
        st.error(f"Clinical trials fetch failed: {str(e)}")
        return pd.DataFrame()

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
    
    return pd.DataFrame(normalized)

def process_news(items: List[Dict], openai_key: str) -> Dict:
    """Categorize and summarize news with GPT-4"""
    if not items:
        return {}
    
    chat = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_key)
    categories = {"Regulatory": [], "Commercialization": [], "Clinical Development": []}
    
    for item in items:
        try:
            response = chat.invoke([
                HumanMessage(f"Categorize:\n{item['title']}\n{item['snippet']}\nOptions: {', '.join(categories.keys())}")
            ])
            category = response.content.strip()
            if category in categories:
                categories[category].append(item)
        except Exception as e:
            st.error(f"News processing failed: {str(e)}")
            continue
    
    for cat, items in categories.items():
        if items:
            try:
                summary = chat.invoke([
                    HumanMessage(f"Summarize these {cat} updates in 3 bullet points: {items}")
                ])
                categories[cat] = summary.content
            except Exception as e:
                st.error(f"Summary generation failed: {str(e)}")
                categories[cat] = "Summary unavailable"
    
    return categories

# --------------------------
# LangGraph Workflow
# --------------------------

def search_node(state: PharmaResearchState) -> PharmaResearchState:
    """Execute search based on type"""
    try:
        clean_query = sanitize_input(state.query) if state.query else ""
        if state.search_type == "news":
            state.news_raw = serper_news_search(
                clean_query,
                state.api_keys["serper"],
                state.time_filter
            )
        else:
            state.clinical_raw = get_clinical_trials(clean_query)
        return state
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return state

def process_node(state: PharmaResearchState) -> PharmaResearchState:
    """Process search results"""
    try:
        if state.search_type == "news":
            state.news_processed = process_news(
                state.news_raw,
                state.api_keys["openai"]
            )
        else:
            state.clinical_processed = {
                "total_trials": len(state.clinical_raw),
                "phases": state.clinical_raw['phase'].value_counts().to_dict(),
                "statuses": state.clinical_raw['status'].value_counts().to_dict()
            }
        return state
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return state

def report_node(state: PharmaResearchState) -> PharmaResearchState:
    """Generate final report"""
    try:
        chat = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, openai_api_key=state.api_keys["openai"])
        
        if state.search_type == "news":
            prompt = f"Generate comprehensive pharmaceutical industry report with markdown formatting based on: {state.news_processed}"
        else:
            prompt = f"Analyze clinical trials data and create professional report: {state.clinical_processed}"
        
        state.report = chat.invoke([HumanMessage(prompt)]).content
        return state
    except Exception as e:
        st.error(f"Report generation failed: {str(e)}")
        return state

# Build workflow
workflow = StateGraph(PharmaResearchState)
workflow.add_node("search_data", search_node)
workflow.add_node("process_data", process_node)
workflow.add_node("generate_report", report_node)

workflow.set_entry_point("search_data")
workflow.add_edge("search_data", "process_data")
workflow.add_edge("process_data", "generate_report")
workflow.add_edge("generate_report", END)

app = workflow.compile()

# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(
    page_title="Pharma Intelligence Suite",
    layout="wide",
    page_icon="🔬"
)

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = PharmaResearchState().dict()

# Load secrets
secrets_available = all(k in st.secrets for k in ["SERPER_API_KEY", "OPENAI_API_KEY"])

# Sidebar Configuration
with st.sidebar:
    st.title("Configuration")
    
    # API Keys from secrets or input
    if secrets_available:
        st.session_state.state['api_keys']['serper'] = st.secrets["SERPER_API_KEY"]
        st.session_state.state['api_keys']['openai'] = st.secrets["OPENAI_API_KEY"]
        st.success("API keys loaded from secrets")
    else:
        st.warning("Using manual API input (store secrets for production)")
        st.session_state.state['api_keys']['serper'] = st.text_input(
            "Serper API Key", 
            type="password",
            value=st.session_state.state['api_keys'].get('serper', '')
        )
        st.session_state.state['api_keys']['openai'] = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=st.session_state.state['api_keys'].get('openai', '')
        )
    
    # Search Parameters
    st.session_state.state['search_type'] = st.radio(
        "Search Type", 
        ["News", "Clinical Trials"],
        index=0 if st.session_state.state.get('search_type') == "news" else 1
    )
    
    if st.session_state.state['search_type'] == "News":
        st.session_state.state['time_filter'] = st.selectbox(
            "Time Filter",
            ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
            index=1
        )

# Main Interface
st.title("Pharma Research Intelligence Platform")

# Search Input
query = st.text_input("Enter search keywords:", key="search_input")
if st.button("Run Analysis"):
    if not all(st.session_state.state['api_keys'].values()):
        st.error("Please provide all API keys")
    else:
        try:
            # Update and validate state
            st.session_state.state['query'] = query
            validated_state = PharmaResearchState(**st.session_state.state)
            
            # Execute workflow
            for step in app.stream(validated_state):
                node_name, node_state = next(iter(step.items()))
                st.session_state.state = node_state.dict()
            
            st.success("Analysis complete!")
            
        except ValidationError as e:
            st.error(f"Invalid data format: {str(e)}")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Display Results
if st.session_state.state.get('report'):
    st.subheader("Analysis Report")
    st.markdown(st.session_state.state['report'])
    
    # Download Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.state['search_type'] == "News":
            st.download_button(
                label="Download News Data",
                data=pd.DataFrame(st.session_state.state['news_raw']).to_csv(),
                file_name="news_results.csv"
            )
    with col2:
        if st.session_state.state['search_type'] == "Clinical Trials":
            st.download_button(
                label="Download Clinical Data",
                data=st.session_state.state['clinical_raw'].to_csv(),
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
    <p>🔒 Secure Processing | 📊 Professional Reporting | 💊 Pharma Intelligence Suite</p>
</div>
""", unsafe_allow_html=True)
