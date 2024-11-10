# # import pickle
# import joblib
# import pandas as pd
# import numpy 
# from sklearn.metrics.pairwise import cosine_similarity

# # Load models data and preprocessed files
# def recommend_models(user_prompt):
#     with open("data/vectorizer.pkl", "rb") as vec_file, open("data/model_matrix.pkl", "rb") as mat_file:
#         vectorizer = joblib.load(vec_file)
#         model_matrix = joblib.load(mat_file)
    
#     df = pd.read_csv("data/models_data.csv")
    
#     # Vectorize user prompt
#     prompt_vector = vectorizer.transform([user_prompt])
    
#     # Calculate similarity scores
#     scores = cosine_similarity(prompt_vector, model_matrix).flatten()
#     top_indices = scores.argsort()[-5:][::-1]
    
#     return df.iloc[top_indices][['model_id', 'description', 'tags']].to_dict(orient='records')
# import joblib
# from sklearn.metrics.pairwise import cosine_similarity

# def recommend_models(user_prompt):
#     # Load the fitted vectorizer and model matrix
#     vectorizer = joblib.load("data/vectorizer.joblib")
#     model_matrix = joblib.load("data/model_matrix.joblib")
    
#     # Transform the user prompt using the fitted vectorizer
#     prompt_vector = vectorizer.transform([user_prompt])
    
#     # Compute cosine similarity between the prompt and model descriptions
#     similarities = cosine_similarity(prompt_vector, model_matrix).flatten()
    
#     # Get indices of the most similar models
#     top_indices = similarities.argsort()[-5:][::-1]  # Top 5 recommendations, in descending order
    
#     # Return recommendations (assuming some logic to fetch model details exists)
#     return [{"model_id": i, "similarity": similarities[i]} for i in top_indices]
# import joblib
# from sklearn.metrics.pairwise import cosine_similarity

# def recommend_models(user_prompt):
#     # Load the fitted vectorizer and model matrix
#     vectorizer = joblib.load("data/vectorizer.joblib")
#     model_matrix = joblib.load("data/model_matrix.joblib")
    
#     # Transform the user prompt using the fitted vectorizer
#     prompt_vector = vectorizer.transform([user_prompt])
    
#     # Compute cosine similarity between the prompt and model descriptions
#     similarities = cosine_similarity(prompt_vector, model_matrix).flatten()
    
#     # Get indices of the most similar models
#     top_indices = similarities.argsort()[-5:][::-1]  # Top 5 recommendations, in descending order
    
#     # Convert numpy data types to Python types before returning
#     recommendations = [
#         {
#             "model_id": int(i),  # Convert numpy.int64 to int
#             "similarity": float(similarities[i])  # Convert numpy.float64 to float
#         }
#         for i in top_indices
#     ]
    
#     return recommendations


import joblib
from sklearn.metrics.pairwise import cosine_similarity

def recommend_models(user_prompt):
    # Load the fitted vectorizer and model matrix
    vectorizer = joblib.load("data/vectorizer.joblib")
    model_matrix = joblib.load("data/model_matrix.joblib")
    
    # Transform the user prompt using the fitted vectorizer
    prompt_vector = vectorizer.transform([user_prompt])
    
    # Compute cosine similarity between the prompt and model descriptions
    similarities = cosine_similarity(prompt_vector, model_matrix).flatten()
    
    # Get indices of the most similar models
    top_indices = similarities.argsort()[-5:][::-1]  # Top 5 recommendations, in descending order
    
    # Mock data for 'description' (replace with actual data if available)
    descriptions = ["Description for model " + str(i) for i in range(len(similarities))]

    # Ensure 'description' key is included in each recommendation
    recommendations = [
        {
            "model_id": int(i),                  # Convert numpy.int64 to int
            "similarity": float(similarities[i]), # Convert numpy.float64 to float
            "description": descriptions[i]        # Mock description
        }
        for i in top_indices
    ]
    
    return recommendations
