import streamlit as st
import requests
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# Configuration
SERPER_API_URL = "https://google.serper.dev/search"
CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_COLUMNS = ['nctId', 'title', 'status', 'phase', 'interventions', 'start_date', 'completion_date']

# --------------------------
# Pydantic State Model
# --------------------------

class PharmaResearchState(BaseModel):
    search_type: str = Field(default="news")
    query: Optional[str] = Field(default=None)
    time_filter: str = Field(default="1 Week")
    news_raw: List[Dict] = Field(default_factory=list)
    clinical_raw: pd.DataFrame = Field(default_factory=lambda: pd.DataFrame(columns=DEFAULT_COLUMNS))
    news_processed: Dict[str, Any] = Field(default_factory=dict)
    clinical_processed: Dict[str, Any] = Field(default_factory=dict)
    report: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

# --------------------------
# Core Functionality
# --------------------------

def sanitize_input(text: str) -> str:
    """Sanitize user input for API queries"""
    return text.strip('"\'').split('(')[0].strip().replace('"', '')

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
        response = requests.post(SERPER_API_URL, headers=headers, json=payload, timeout=15)
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
            response = requests.get(CLINICAL_TRIALS_API, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            studies = data.get('studies', [])
            if not studies:
                break
            all_studies.extend(studies)
            if not data.get('nextPageToken'):
                break
            params['pageToken'] = data['nextPageToken']
    except Exception as e:
        st.error(f"Clinical trials fetch failed: {str(e)}")
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

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
    
    return pd.DataFrame(normalized) if normalized else pd.DataFrame(columns=DEFAULT_COLUMNS)

def process_news(items: List[Dict], openai_key: str) -> Dict:
    """Categorize and summarize news with GPT-4"""
    if not items:
        return {}
    
    try:
        chat = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_key)
        categories = {
            "Regulatory": [],
            "Commercialization": [],
            "Clinical Development": []
        }
        
        for item in items:
            try:
                response = chat.invoke([
                    HumanMessage(f"Categorize:\n{item['title']}\n{item['snippet']}\nOptions: {', '.join(categories.keys())}")
                ])
                category = response.content.strip()
                if category in categories:
                    categories[category].append(item)
            except Exception as e:
                st.error(f"News categorization failed: {str(e)}")
                continue
        
        for cat, items in categories.items():
            if items:
                try:
                    summary = chat.invoke([
                        HumanMessage(f"Summarize these {cat} updates in 3 bullet points: {items}")
                    ])
                    categories[cat] = summary.content
                except Exception as e:
                    st.error(f"Summary generation failed for {cat}: {str(e)}")
                    categories[cat] = "Summary unavailable"
        
        return categories
    except Exception as e:
        st.error(f"News processing failed: {str(e)}")
        return {}

# --------------------------
# LangGraph Workflow
# --------------------------

def search_node(state: PharmaResearchState) -> dict:
    """Execute search based on type"""
    try:
        clean_query = sanitize_input(state.query) if state.query else ""
        if not clean_query:
            st.error("Please enter a valid search query")
            return state.model_dump()
        
        new_state = state.model_dump()
        
        if state.search_type == "news":
            new_state['news_raw'] = serper_news_search(
                clean_query,
                st.secrets["SERPER_API_KEY"],
                state.time_filter
            )
        else:
            clinical_data = get_clinical_trials(clean_query)
            new_state['clinical_raw'] = clinical_data.to_dict(orient='list')
        
        return new_state
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return state.model_dump()

def process_node(state: dict) -> dict:
    """Process search results"""
    try:
        new_state = state.copy()
        
        if state['search_type'] == "news":
            new_state['news_processed'] = process_news(
                state['news_raw'],
                st.secrets["OPENAI_API_KEY"]
            )
        else:
            clinical_df = pd.DataFrame(state['clinical_raw'])
            new_state['clinical_processed'] = {
                "total_trials": len(clinical_df),
                "phases": clinical_df['phase'].value_counts().to_dict(),
                "statuses": clinical_df['status'].value_counts().to_dict()
            }
        
        return new_state
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return state

def report_node(state: dict) -> dict:
    """Generate final report"""
    try:
        new_state = state.copy()
        chat = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, openai_api_key=st.secrets["OPENAI_API_KEY"])
        
        if state['search_type'] == "news":
            prompt = f"Generate pharmaceutical industry report with markdown formatting based on: {state['news_processed']}"
        else:
            prompt = f"Analyze clinical trials data and create professional report: {state['clinical_processed']}"
        
        response = chat.invoke([HumanMessage(prompt)])
        new_state['report'] = response.content
        
        return new_state
    except Exception as e:
        st.error(f"Report generation failed: {str(e)}")
        return state

# Build workflow
workflow = StateGraph(PharmaResearchState)
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

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = PharmaResearchState().model_dump()

# Sidebar Configuration
with st.sidebar:
    st.title("Configuration")
    
    search_type = st.radio(
        "Search Type", 
        ["News", "Clinical Trials"],
        index=0 if st.session_state.state.get('search_type') == "news" else 1
    )
    st.session_state.state['search_type'] = search_type
    
    if search_type == "News":
        time_filter = st.selectbox(
            "Time Filter",
            ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
            index=1
        )
        st.session_state.state['time_filter'] = time_filter

# Main Interface
st.title("Pharma Research Intelligence Platform")

query = st.text_input("Enter search keywords:", key="search_input")
if st.button("Run Analysis"):
    st.session_state.state['query'] = query
    
    try:
        # Validate input state
        validated_state = PharmaResearchState(**st.session_state.state)
        
        # Execute workflow
        final_state = None
        for output in app.stream(validated_state):
            final_state = output
        
        if final_state:
            st.session_state.state = final_state.model_dump()
            st.success("Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.session_state.state = PharmaResearchState().model_dump()

# Display Results
if st.session_state.state.get('report'):
    st.subheader("Analysis Report")
    st.markdown(st.session_state.state['report'])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.state['search_type'] == "News":
            st.download_button(
                "Download News Data",
                pd.DataFrame(st.session_state.state['news_raw']).to_csv(),
                "news_results.csv"
            )
    with col2:
        if st.session_state.state['search_type'] == "Clinical Trials":
            st.download_button(
                "Download Clinical Data",
                pd.DataFrame(st.session_state.state['clinical_raw']).to_csv(),
                "clinical_trials.csv"
            )

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
