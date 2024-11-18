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
                    st.write(f"**Model ID:** {rec.get('model_id', 'N/A')}")
                    st.write(f"**Model Name:** {rec.get('model_name', 'N/A')}")
                    st.write(f"**Description:** {rec.get('description', 'N/A')}")
                    st.write(f"**Tags:** {rec.get('tags', 'N/A')}")
                    st.write(f"**Model Downloads:** {rec.get('downloads', 'N/A')}")
                    st.write(f"**Likes:** {rec.get('likes', 'N/A')}")
                    st.write(f"**Similarity Score:** {rec.get('similarity', 'N/A'):.4f}")
                    st.write("---")
            else:
                st.error("Failed to retrieve recommendations. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt to get recommendations.")
