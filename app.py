import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

import pandas as pd
import numpy as np
import pickle
import requests

# -------------------------
# Load Resources
# -------------------------

@st.cache_data
def load_resources():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    npz_file = np.load("cosine_similarity_matrix.npz", allow_pickle=True)
    cosine_sim = npz_file["cosine_sim"]

    movie_data = pd.read_csv("movie_data.csv")
    title_to_index = pd.Series(movie_data.index, index=movie_data['title'].str.lower())

    return tfidf_vectorizer, cosine_sim, title_to_index, movie_data

tfidf_vectorizer, cosine_sim, title_to_index, movie_data = load_resources()

# -------------------------
# Recommend Function
# -------------------------

def recommend_movies(title, num_recommendations=5):
    title = title.lower()

    if title not in title_to_index:
        return ["Movie not found. Please try another title."], []

    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    recommended_titles = [movie_data.iloc[i[0]]['title'] for i in sim_scores]
    return recommended_titles, [get_movie_poster(t) for t in recommended_titles]

# -------------------------
# OMDb Poster Fetching
# -------------------------

OMDB_API_KEY = "122ba293"  # 🔑 Replace this with your real API key

def get_movie_poster(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data.get("Response") == "True":
        return data.get("Poster")
    return None

# -------------------------
# Custom CSS for Styling
# -------------------------

st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .title {
            color: #dc2626;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
        }
        .recommendation {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .poster {
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Streamlit UI
# -------------------------

st.markdown('<p class="title">🎬 Movie Recommender System</p>', unsafe_allow_html=True)

movie_list = movie_data['title'].values
selected_movie = st.selectbox("Choose a movie to get recommendations:", movie_list)

if st.button("Recommend"):
    recommendations, posters = recommend_movies(selected_movie)

    if "Movie not found" in recommendations[0]:
        st.error(recommendations[0])
    else:
        st.subheader("Recommended Movies:")
        for i, (rec, poster) in enumerate(zip(recommendations, posters), 1):
            with st.container():
                cols = st.columns([1, 4])
                if poster:
                    cols[0].image(poster, width=100, caption=None, use_column_width=False)
                else:
                    cols[0].write("🎞️")
                cols[1].markdown(f"<div class='recommendation'><h5>{i}. {rec}</h5></div>", unsafe_allow_html=True)
