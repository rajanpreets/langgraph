import streamlit as st
from workflow import app
import session_state
from workflow import app
from tools.clinical_trials import get_clinical_trials_data  # Explicit import
from tools.news_tools import serper_news_search, analyze_news  # Explicit import

# Initialize session
if 'state' not in st.session_state:
    st.session_state.state = {
        "api_keys": {"serper": "", "openai": ""},
        "keywords": [],
        "search_type": "news"
    }

# UI Layout
st.set_page_config(page_title="Pharma Research Suite", layout="wide")

# Sidebar Controls
with st.sidebar:
    st.header("Configuration")
    st.session_state.state["api_keys"]["serper"] = st.text_input(
        "Serper API Key", type="password")
    st.session_state.state["api_keys"]["openai"] = st.text_input(
        "OpenAI API Key", type="password")
    st.session_state.state["search_type"] = st.selectbox(
        "Search Type", ["news", "clinical"])

# Main Interface
st.header("Pharma Research Intelligence Platform")

# Search Input
keywords = st.text_input("Enter keywords (comma-separated)")
if keywords:
    st.session_state.state["keywords"] = [k.strip() for k in keywords.split(",")]

# Execute Workflow
if st.button("Run Analysis"):
    if not all(st.session_state.state["api_keys"].values()):
        st.error("Please provide all API keys")
    else:
        # Execute LangGraph workflow
        for step in app.stream(st.session_state.state):
            node = list(step.keys())[0]
            st.write(f"✅ Completed step: {node}")
        
        # Display results
        report = st.session_state.state.get("report")
        if report.get("news"):
            st.subheader("News Analysis")
            st.json(report["news"])
        
        if report.get("clinical"):
            st.subheader("Clinical Trials Analysis")
            st.json(report["clinical"])
