import requests
import pandas as pd
from dateutil.parser import parse, ParserError

def get_clinical_trials_data(keyword: str) -> pd.DataFrame:
    """Fetch clinical trials data for a keyword"""
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.term": keyword, "pageSize": 1000}
    
    all_studies = {"studies": []}
    page_token = None
    
    while True:
        params["pageToken"] = page_token
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            all_studies["studies"].extend(data.get("studies", []))
            page_token = data.get("nextPageToken")
            if not page_token: break
        else:
            raise Exception(f"API Error: {response.status_code}")
    
    # Normalization and processing logic from previous implementation
    # ... (include full normalization code from earlier example) ...
    
    return pd.DataFrame(normalized_data)

def process_clinical_trials(df: pd.DataFrame) -> dict:
    """Process clinical trials data into structured format"""
    return {
        "total_trials": len(df),
        "phase_distribution": df['phases'].value_counts().to_dict(),
        "status_distribution": df['overallStatus'].value_counts().to_dict(),
        "interventions": df['interventionDrug'].explode().value_counts().head(5).to_dict()
    }
