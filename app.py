import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------
# Load Resources
# -------------------------

@st.cache_data
def load_resources():
    # Load TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    # Load cosine similarity matrix
    npz_file = np.load("cosine_similarity_matrix.npz", allow_pickle=True)
    cosine_sim = npz_file["cosine_sim"]

    # Load movie dataset
    movie_data = pd.read_csv("movies.csv")

    # Build title to index mapping
    title_to_index = pd.Series(movie_data.index, index=movie_data['title'].str.lower())

    return tfidf_vectorizer, cosine_sim, title_to_index, movie_data

tfidf_vectorizer, cosine_sim, title_to_index, movie_data = load_resources()

# -------------------------
# Recommend Function
# -------------------------

def recommend_movies(title, num_recommendations=5):
    title = title.lower()

    if title not in title_to_index:
        return ["Movie not found. Please try another title."]

    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # skip self

    recommended_titles = [movie_data.iloc[i[0]]['title'] for i in sim_scores]
    return recommended_titles

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender System")

movie_list = movie_data['title'].values
selected_movie = st.selectbox("Choose a movie to get recommendations:", movie_list)

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)
    st.subheader("Recommended Movies:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
