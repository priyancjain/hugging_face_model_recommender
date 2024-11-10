import requests
import pandas as pd

def fetch_model_data():
    # Hugging Face API endpoint
    url = "https://huggingface.co/api/models"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        models = []
        
        for model in data:
            models.append({
                "model_id": model.get("modelId"),
                "description": model.get("pipeline_tag", ""),
                "tags": ", ".join(model.get("tags", [])),
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "language": model.get("languages", "unknown")
            })
            
        df = pd.DataFrame(models)
        df.to_csv("data/models_data.csv", index=False)
        print("Model data saved to data/models_data.csv")
    else:
        print("Failed to fetch model data")

# Run this function periodically to update data
if __name__ == "__main__":
    fetch_model_data()
