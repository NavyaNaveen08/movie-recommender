import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import gdown

# Google Drive file download function
def download_file_from_gdrive(file_url, output_file):
    gdown.download(file_url, output_file, quiet=False)
    return output_file

# Download and load resources
@st.cache_resource
def load_resources():
    # Download required files from Google Drive
    tfidf_vectorizer_file = download_file_from_gdrive('https://drive.google.com/uc?export=download&id=FILE_ID', 'tfidf_vectorizer.pkl')
    cosine_sim_file = download_file_from_gdrive('https://drive.google.com/uc?export=download&id=FILE_ID', 'cosine_similarity_matrix.npy')
    title_to_index_file = download_file_from_gdrive('https://drive.google.com/uc?export=download&id=FILE_ID', 'title_to_index.csv')
    movie_data_file = download_file_from_gdrive('https://drive.google.com/uc?export=download&id=FILE_ID', 'movie_data.csv')

    # Load the files
    with open(tfidf_vectorizer_file, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    cosine_sim = np.load(cosine_sim_file)
    title_to_index = pd.read_csv(title_to_index_file, index_col=0).squeeze("columns")
    movie_data = pd.read_csv(movie_data_file)
    
    return tfidf_vectorizer, cosine_sim, title_to_index, movie_data

# Load resources
tfidf_vectorizer, cosine_sim, title_to_index, movie_data = load_resources()

# Fetch movie posters
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_TMDB_API_KEY"
    response = requests.get(url)
    data = response.json()
    try:
        return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    except KeyError:
        return None

# Movie recommendation function
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
        movie_id = movie_data.iloc[i[0]].id
        poster = fetch_poster(movie_id)
        recommended.append((movie_title, poster))
    
    return recommended

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Custom header styling
st.markdown("""
    <style>
    .title { font-size: 40px; font-weight: bold; color: #FF6F61; text-align: center; margin-bottom: 30px; }
    .subtitle { font-size: 20px; text-align: center; margin-bottom: 30px; }
    .movie-title { font-size: 18px; color: #FF6F61; font-weight: bold; text-align: center; }
    </style>
    <div class="title">🎬 Movie Recommendation System</div>
    <div class="subtitle">Find similar movies based on your favorite!</div>
""", unsafe_allow_html=True)

# User input
movie_input = st.text_input("Enter a movie title:", "Avatar")

# Recommendation button
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
