# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import joblib

# def load_and_preprocess_data():
#     # Load the dataset
#     df = pd.read_csv("data/models_data.csv")
    
#     # Combine the description and tags columns, handling any NaN values
#     text_data = (df['description'].fillna('') + " " + df['tags'].fillna('')).values
    
#     # Initialize and fit the TF-IDF vectorizer on the text data
#     vectorizer = TfidfVectorizer(stop_words='english')
#     model_matrix = vectorizer.fit_transform(text_data)  # Fit is done here
    
#     # Save the vectorizer and model matrix
#     joblib.dump(vectorizer, "data/vectorizer.joblib")
#     joblib.dump(model_matrix, "data/model_matrix.joblib")
#     print("Data preprocessing complete and saved.")

# # Run the preprocessing step to save the vectorizer and model matrix
# if __name__ == "__main__":
#     load_and_preprocess_data()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_and_preprocess_data():
    # Load the dataset with all model details
    df = pd.read_csv("data/models_data.csv")
    
    # Combine text fields for vectorization
    text_data = (df['description'].fillna('') + " " + df['tags'].fillna('')).values
    
    # Initialize and fit the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    model_matrix = vectorizer.fit_transform(text_data)
    
    # Save vectorizer, model matrix, and model details
    joblib.dump(vectorizer, "data/vectorizer.joblib")
    joblib.dump(model_matrix, "data/model_matrix.joblib")
    joblib.dump(df.to_dict(orient="records"), "data/models_data.joblib")  # Save all model info
    
    print("Data preprocessing complete and saved.")

if __name__ == "__main__":
    load_and_preprocess_data()
