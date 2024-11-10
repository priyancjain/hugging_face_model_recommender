# frontend/app.py

# import streamlit as st
# import requests
# import sys
# from pathlib import Path

# # Add the src directory to sys.path to enable absolute imports
# sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# # Now you can import recommend_models from recommender.py in src
# from recommender import recommend_models

# st.title("Model Recommender")
# st.write("Enter a prompt to get model recommendations based on your task.")

# # Input field for the user prompt
# user_prompt = st.text_input("Describe your task:")

# # Button to get recommendations
# if st.button("Get Recommendations"):
#     if user_prompt:
#         # Send a POST request to the FastAPI backend
#         try:
#             response = requests.post(
#                 "http://127.0.0.1:8000/recommend_models/",
#                 json={"user_prompt": user_prompt},
#             )
            
#             if response.status_code == 200:
#                 recommendations = response.json()
#                 st.write("Recommended Models:")
#                 for rec in recommendations:
#                     st.write(f"**Model ID:** {rec['model_id']}")
#                     st.write(f"**Description:** {rec['description']}")
#                     st.write(f"**Tags:** {rec['tags']}")
#                     st.write("---")
#             else:
#                 st.error("Failed to retrieve recommendations. Please try again.")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error: {e}")
#     else:
#         st.warning("Please enter a prompt to get recommendations.")
# frontend/app.py

import streamlit as st
import requests

st.title("Model Recommender")
st.write("Enter a prompt to get model recommendations based on your task.")

# Input field for the user prompt
user_prompt = st.text_input("Describe your task:")

# Button to get recommendations
if st.button("Get Recommendations"):
    if user_prompt:
        # Send a POST request to the FastAPI backend
        try:
            response = requests.post(
                "http://127.0.0.1:8000/recommend_models/",
                json={"user_prompt": user_prompt},
            )
            
            if response.status_code == 200:
                recommendations = response.json()
                st.write("### Recommended Models:")
                for rec in recommendations:
                    st.write(f"**Model ID:** {rec['model_id']}")
                    st.write(f"**Similarity Score:** {rec['similarity']:.4f}")
                    st.write(f"**Description:** {rec['description']}")
                    st.write("---")
            else:
                st.error("Failed to retrieve recommendations. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt to get recommendations.")
