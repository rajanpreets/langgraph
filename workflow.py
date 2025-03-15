from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain.schema import AIMessage

# Local imports
from news_tools import serper_news_search, analyze_news
from clinical_trials import get_clinical_trials_data, process_clinical_trials

def workflow(state: Dict[str, Any]) -> Dict[str, Any]:
    """Main workflow decision function"""
    search_type = state.get("search_type", "news")
    api_keys = state["api_keys"]
    keywords = state["keywords"]
    
    if search_type == "news":
        news_results = {}
        for keyword in keywords:
            news_results[keyword] = serper_news_search(
                keyword, api_keys["serper"])
        state["news_raw"] = news_results
        
    elif search_type == "clinical":
        clinical_results = {}
        for keyword in keywords:
            trials_df = get_clinical_trials_data(keyword)
            clinical_results[keyword] = process_clinical_trials(trials_df)
        state["clinical_raw"] = clinical_results
        
    return state

def process_news(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process news results"""
    state["news_processed"] = {
        k: analyze_news(v, state["api_keys"]["openai"])
        for k, v in state["news_raw"].items()
    }
    return state

def generate_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final report"""
    report = {"news": {}, "clinical": {}}
    
    if "news_processed" in state:
        report["news"] = state["news_processed"]
    
    if "clinical_raw" in state:
        report["clinical"] = state["clinical_raw"]
    
    state["report"] = report
    state["messages"] = [AIMessage(content=str(report))]
    return state

# Create workflow graph
workflow_graph = StateGraph(dict)
workflow_graph.add_node("workflow", workflow)
workflow_graph.add_node("process_news", process_news)
workflow_graph.add_node("generate_report", generate_report)

# Define edges
workflow_graph.set_entry_point("workflow")
workflow_graph.add_conditional_edges(
    "workflow",
    lambda state: "process_news" if state.get("search_type") == "news" else "generate_report"
)
workflow_graph.add_edge("process_news", "generate_report")
workflow_graph.add_edge("generate_report", END)

app = workflow_graph.compile()
