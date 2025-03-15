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
# ... (keep previous imports and configuration)

# --------------------------
# State Management
# --------------------------

class PharmaResearchState(BaseModel):
    """Pydantic model for state validation"""
    api_keys: Dict[str, str] = Field(default_factory=dict)
    search_type: str = "news"
    query: Optional[str] = None
    news_raw: List[Dict] = Field(default_factory=list)
    clinical_raw: pd.DataFrame = Field(default_factory=pd.DataFrame)
    report: Optional[str] = None

def init_state() -> PharmaResearchState:
    """Initialize validated state"""
    return PharmaResearchState(
        api_keys=st.session_state.get('api_keys', {'serper': '', 'openai': ''}),
        search_type=st.session_state.get('search_type', 'news'),
        query=st.session_state.get('query', ''),
        news_raw=st.session_state.get('news_raw', []),
        clinical_raw=st.session_state.get('clinical_raw', pd.DataFrame()),
        report=st.session_state.get('report', None)
    )

# --------------------------
# Modified Workflow Nodes
# --------------------------

def search_node(state: PharmaResearchState) -> PharmaResearchState:
    """Execute search based on type"""
    if state.search_type == 'news':
        state.news_raw = serper_news_search(
            state.query,
            state.api_keys['serper'],
            state.time_filter
        )
    else:
        state.clinical_raw = get_clinical_trials(state.query)
    return state

def process_node(state: PharmaResearchState) -> PharmaResearchState:
    """Process search results"""
    if state.search_type == 'news':
        state.report = process_news(
            state.news_raw,
            state.api_keys['openai']
        )
    else:
        state.report = analyze_clinical_trials(state.clinical_raw)
    return state

# --------------------------
# Updated Streamlit UI
# --------------------------

if st.button("Run Analysis"):
    if not all(st.session_state.api_keys.values()):
        st.error("Please provide all API keys")
    else:
        try:
            # Initialize validated state
            state = init_state()
            state.query = query
            state.search_type = st.session_state.search_type.lower()
            
            # Execute workflow
            for step in app.stream(state):
                node_name = list(step.keys())[0]
                st.session_state.state = step[node_name].dict()
            
            st.session_state.report = st.session_state.state['report']
            
        except ValidationError as e:
            st.error(f"Validation error: {str(e)}")
            st.stop()
