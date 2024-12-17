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
    
#     # Mock data for 'description' (replace with actual data if available)
#     descriptions = ["Description for model " + str(i) for i in range(len(similarities))]

#     # Ensure 'description' key is included in each recommendation
#     recommendations = [
#         {
#             "model_id": int(i),                  # Convert numpy.int64 to int
#             "similarity": float(similarities[i]), # Convert numpy.float64 to float
#             "description": descriptions[i]        # Mock description
#         }
#         for i in top_indices
#     ]
    
#     return recommendations
# import joblib
# from sklearn.metrics.pairwise import cosine_similarity

# def recommend_models(user_prompt):
#     # Load vectorizer, model matrix, and model data
#     vectorizer = joblib.load("data/vectorizer.joblib")
#     model_matrix = joblib.load("data/model_matrix.joblib")
#     model_data = joblib.load("data/models_data.joblib")  # List of dictionaries with model info
    
#     # Transform the user prompt using the fitted vectorizer
#     prompt_vector = vectorizer.transform([user_prompt])
    
#     # Compute cosine similarity between the prompt and model descriptions
#     similarities = cosine_similarity(prompt_vector, model_matrix).flatten()
    
#     # Get indices of the most similar models
#     top_indices = similarities.argsort()[-5:][::-1]  # Top 5 recommendations, descending order

#     # Include full model information for each recommendation
#     # recommendations = [
#     #     {
#     #         "similarity": float(similarities[i]),  # Ensure compatibility with JSON serialization
#     #         **model_data[i]  # Unpack all model details for the selected model
#     #     }
#     #     for i in top_indices
#     # ]
#     recommendations = [
#     {
#         "model_id": int(i),
#         "similarity": float(similarities[i]),
#         "model_name": model_data[i].get("model_id", "Unknown Model"),
#         "description": model_data[i].get("description", "No description available"),
#         "tags": model_data[i].get("tags", "No tags available"),
#         "downloads": model_data[i].get("downloads", "N/A"),
#         "likes": model_data[i].get("likes", "N/A")
#     }
#     for i in top_indices
# ]

#     return recommendations
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def clean_data(data):
    """
    Recursively clean the data to replace NaN, inf, or -inf with valid values.
    """
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None  # Replace invalid float values with None
    return data


def recommend_models(user_prompt):
    # Load the vectorizer and model data
    vectorizer = joblib.load("data/vectorizer.joblib")
    model_matrix = joblib.load("data/model_matrix.joblib")
    model_data = joblib.load("data/models_data.joblib")  # List of dictionaries with model details

    # Transform the user prompt using the vectorizer
    prompt_vector = vectorizer.transform([user_prompt])

    # Compute cosine similarity between the prompt vector and the model matrix
    similarities = cosine_similarity(prompt_vector, model_matrix).flatten()

    # Get the indices of the top 5 most similar models
    top_indices = similarities.argsort()[-5:][::-1]

    # Prepare the recommendation data
    recommendations = [
        {
             "model_id": int(i),
            "similarity": float(similarities[i]),
            "model_name": model_data[i].get("model_id", "Unknown Model"),
            "description": model_data[i].get("description", "No description available"),
            "tags": model_data[i].get("tags", "No tags available"),
            "downloads": model_data[i].get("downloads", "N/A"),
            "likes": model_data[i].get("likes", "N/A")
        }
        for i in top_indices
    ]

    # Clean the recommendation data
    return clean_data(recommendations)
