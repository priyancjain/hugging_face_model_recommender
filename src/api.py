from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .recommender import recommend_models

app = FastAPI()

# Configure CORS to allow requests from Streamlit
origins = ["http://localhost:8501", "http://127.0.0.1:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected data model
class RecommendationRequest(BaseModel):
    user_prompt: str

@app.post("/recommend_models/")
async def get_recommendations(request: RecommendationRequest):
    recommendations = recommend_models(request.user_prompt)
    return recommendations
