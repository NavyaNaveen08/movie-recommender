import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# Load the saved model files
@st.cache_resource
def load_resources():
    tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    cosine_sim = np.load("cosine_similarity_matrix.npy")
    title_to_index = pd.read_csv("title_to_index.csv", index_col=0).squeeze("columns")
    movie_data = pd.read_csv("movie_data.csv")
    return tfidf_vectorizer, cosine_sim, title_to_index, movie_data

tfidf_vectorizer, cosine_sim, title_to_index, movie_data = load_resources()

# Function to fetch movie posters
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_TMDB_API_KEY"
    response = requests.get(url)
    data = response.json()
    try:
        return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    except KeyError:
        return None

# Recommend function
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

# Header with custom styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #FF6F61;
        text-align: center;
        margin-bottom: 30px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        margin-bottom: 30px;
    }
    .movie-title {
        font-size: 18px;
        color: #FF6F61;
        font-weight: bold;
        text-align: center;
    }
    </style>
    <div class="title">🎬 Movie Recommendation System</div>
    <div class="subtitle">Find similar movies based on your favorite!</div>
""", unsafe_allow_html=True)

# Input field
movie_input = st.text_input("Enter a movie title:", "Avatar")

# Button and recommendation
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
import gdown
import pickle

# Google Drive file URL
file_url = 'https://drive.google.com/uc?export=download&id=1nqwU56EgKh2NO6H_Sf2Lg9XCAymofbCY'  # Replace FILE_ID with your actual file ID

# Download the cosine similarity matrix
output_file = 'cosine_similarity_matrix.pkl'
gdown.download(file_url, output_file, quiet=False)

# Load the cosine similarity matrix (assuming it's pickled)
with open(output_file, 'rb') as f:
    cosine_similarity_matrix = pickle.load(f)

# Now you can use the cosine_similarity_matrix in your recommender system
