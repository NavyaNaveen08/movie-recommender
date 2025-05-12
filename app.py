import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------
# Load resources
# -----------------------
@st.cache_data
def load_resources():
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
    data = pd.read_csv("movies.csv")
    npz_file = np.load("cosine_similarity_matrix.npz")
    cosine_sim = npz_file["cosine_sim"]
    title_to_index = pd.Series(data.index, index=data["title"].str.lower())
    return tfidf_vectorizer, cosine_sim, title_to_index, data

# -----------------------
# Recommend movies
# -----------------------
def recommend_movies(title, cosine_sim, title_to_index, data, top_n=10):
    title = title.lower()
    if title not in title_to_index:
        return []

    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return data["title"].iloc[movie_indices].tolist()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender System")
st.markdown("Enter a movie title to get similar recommendations!")

# Load data
tfidf_vectorizer, cosine_sim, title_to_index, data = load_resources()

# User input
movie_input = st.text_input("Enter a movie title")

if movie_input:
    recommendations = recommend_movies(movie_input, cosine_sim, title_to_index, data)
    if recommendations:
        st.subheader("Recommended Movies:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("Sorry, that movie was not found. Please check the spelling or try another title.")
