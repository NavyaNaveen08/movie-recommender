import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# -----------------------
# Download from GitHub
# -----------------------
def download_from_github(url, local_filename):
    response = requests.get(url)
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    return local_filename

# -----------------------
# Load All Resources
# -----------------------
@st.cache_resource
def load_resources():
    base_url = "https://raw.githubusercontent.com/NavyaNaveen08/movie-recommender/main/"
    
    # Files from GitHub
    tfidf_vectorizer_file = download_from_github(base_url + 'tfidf_vectorizer.pkl', 'tfidf_vectorizer.pkl')
    title_to_index_file = download_from_github(base_url + 'title_to_index.csv', 'title_to_index.csv')
    movie_data_file = download_from_github(base_url + 'movie_data.csv', 'movie_data.csv')
    cosine_sim_file = download_from_github(base_url + 'cosine_similarity_matrix.npz', 'cosine_similarity_matrix.npz')

    # Load contents
    with open(tfidf_vectorizer_file, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    cosine_sim = np.load(cosine_sim_file)['cosine_sim']
    title_to_index = pd.read_csv(title_to_index_file, index_col=0).squeeze("columns")
    movie_data = pd.read_csv(movie_data_file)
    
    return tfidf_vectorizer, cosine_sim, title_to_index, movie_data

# -----------------------
# Fetch Poster from OMDb
# -----------------------
def fetch_poster(movie_id):
    url = f"http://www.omdbapi.com/?i={movie_id}&apikey=122ba293"
    response = requests.get(url)
    data = response.json()
    
    if data.get('Response') == 'True':
        return data.get('Poster', None)
    else:
        return None

# -----------------------
# Recommend Movies
# -----------------------
def recommend(movie):
    movie = movie.lower()
    if movie not in title_to_index:
        return []
    
    index = title_to_index[movie]
    distances = cosine_sim[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended = []
    for i in movie_list:
        movie_title = movie_data.iloc[i[0]].title
        movie_imdb_id = movie_data.iloc[i[0]].imdb_id
        poster = fetch_poster(movie_imdb_id)
        recommended.append((movie_title, poster))
    
    return recommended

# -----------------------
# Streamlit UI
# -----------------------
tfidf_vectorizer, cosine_sim, title_to_index, movie_data = load_resources()
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .title { font-size: 40px; font-weight: bold; color: #FF6F61; text-align: center; margin-bottom: 30px; }
    .subtitle { font-size: 20px; text-align: center; margin-bottom: 30px; }
    .movie-title { font-size: 18px; color: #FF6F61; font-weight: bold; text-align: center; }
    </style>
    <div class="title">🎬 Movie Recommendation System</div>
    <div class="subtitle">Find similar movies based on your favorite!</div>
""", unsafe_allow_html=True)

movie_input = st.text_input("Enter a movie title:", "Avatar")

if st.button("Get Recommendations"):
    if movie_input.strip() == "":
        st.warning("Please enter a valid movie title.")
    else:
        with st.spinner('Fetching recommendations...'):
            results = recommend(movie_input)
            if results:
                st.success("Here are some recommendations:")
                cols = st.columns(5)
                for i, (title, poster) in enumerate(results):
                    with cols[i % 5]:
                        st.image(poster, caption=title, use_column_width=True)
                        st.markdown(f"<p class='movie-title'>{title}</p>", unsafe_allow_html=True)
            else:
                st.error("Sorry, movie not found in our dataset.")
