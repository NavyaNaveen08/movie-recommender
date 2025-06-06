import streamlit as st

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

import pandas as pd
import numpy as np
import pickle
import requests

# -------------------------

st.markdown("""
    <style>
        .stApp {
            background-color: #0f0f0f;
            color: #ffffff;
        }
        .big-title {
            color: #ef4444;
            text-align: center;
            font-size: 40px; 
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 30px;
        }
        .recommendation {
            background-color: #1c1c1c;
            padding: 10px;
            border-radius: 12px;
            margin-bottom: 15px;
            color: #ffffff;
        }
        .movie-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .movie-desc {
            font-size: 16px;
            color: #dddddd;
        }
        .stSelectbox label {
            color: #ffffff;
        }
        .stButton>button {
            background-color: black;
            color: red;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            margin-top: 15px;
        }
        .stButton>button:hover {
            background-color: #b91c1c;
        }
        .poster {
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(255, 0, 0, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

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
# OMDb API Key (only for posters and plot descriptions)
# -------------------------

OMDB_API_KEY = "122ba293" 

@st.cache_data(show_spinner=False)
def fetch_movie_details(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        response = requests.get(url).json()
        if response.get("Response") == "True":
            poster = response.get("Poster")
            plot = response.get("Plot", "Description not available.")
            return poster, plot
    except:
        pass
    return None, "Description not available."

# -------------------------
# Recommend Function
# -------------------------

def recommend_movies(title, num_recommendations=5):
    title = title.lower()

    if title not in title_to_index:
        return ["Movie not found. Please try another title."], [], []

    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    recommended_titles = []
    posters = []
    descriptions = []

    for i in sim_scores:
        movie_row = movie_data.iloc[i[0]]
        recommended_titles.append(movie_row['title'])
        poster, description = fetch_movie_details(movie_row['title'])
        posters.append(poster)
        descriptions.append(description)

    return recommended_titles, posters, descriptions

# -------------------------
# Streamlit UI
# -------------------------

st.markdown('<div class="big-title">🎬 Movie Recommender System</div>', unsafe_allow_html=True)

movie_list = movie_data['title'].values
selected_movie = st.selectbox("Choose a movie to get recommendations:", movie_list)

if st.button("Recommend"):
    with st.spinner("Fetching recommendations..."):
        recommendations, posters, descriptions = recommend_movies(selected_movie)

    if "Movie not found" in recommendations[0]:
        st.error(recommendations[0])
    else:
        st.subheader("Recommended Movies:")
        for i, (rec, poster, desc) in enumerate(zip(recommendations, posters, descriptions), 1):
            with st.container():
                cols = st.columns([1, 4])
                if poster:
                    cols[0].image(poster, width=100, use_container_width=True)
                else:
                    cols[0].write("🎞️")
                cols[1].markdown(f"""
                    <div class='recommendation'>
                        <div class='movie-title'>{i}. {rec}</div>
                        <div class='movie-desc'>{desc}</div>
                    </div>
                """, unsafe_allow_html=True)
