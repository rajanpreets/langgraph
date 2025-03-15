import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

def serper_news_search(keyword: str, api_key: str, time_filter: str = "qdr:7w") -> list:
    """Search news using Serper API"""
    payload = {
        "q": keyword,
        "tbm": "nws",
        "tbs": time_filter,
        "num": 5
    }
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    
    response = requests.post("https://google.serper.dev/search", 
                           json=payload, 
                           headers=headers)
    return response.json().get("newsResults", [])

def analyze_news(items: list, api_key: str) -> dict:
    """Categorize and summarize news articles"""
    chat_model = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=api_key)
    categories = {"Regulatory": [], "Commercialization": [], "Clinical": []}
    
    for item in items:
        response = chat_model.invoke([
            HumanMessage(content=f"""Categorize this news:
            {item['title']} - {item['snippet']}
            Options: Regulatory, Commercialization, Clinical
            Respond only with category name.""")
        ])
        category = response.content.strip()
        if category in categories:
            categories[category].append(item)
    
    # Generate summaries
    for category, items in categories.items():
        if items:
            summary_prompt = f"Summarize these {category} updates in 3 bullet points"
            response = chat_model.invoke([HumanMessage(content=summary_prompt)])
            categories[category] = response.content
            
    return categories
