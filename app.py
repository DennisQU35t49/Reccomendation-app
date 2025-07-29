# Import all the necessary libraries
import streamlit as st
import pandas as pd
import pickle
import re


# Load the the pickle components

# Loading in the vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Loading in the tfidf matrix
with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# Loading in the nn_model
with open("content_nn.pkl", "rb") as f:
    nn_model = pickle.load(f)

#load in the data (unique_books)
books_df = pd.read_pickle("unique_books.pkl")

# Clean the title
def preprocess_text(text):
    text = str(text).lower() # Converts the title to string and lower case
    text = re.sub(r'[^a-z0-9\s]', '', text) # Removes foreign charachters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Recommendation function
def recommend(title, books_df, vectorizer, nn_model, tfidf_matrix, n_neighbors=5):
    clean_title = preprocess_text(title)

    if 'clean_title' not in books_df.columns:
        books_df['clean_title'] = books_df['original_title'].apply(preprocess_text)

    match = books_df[books_df['clean_title'] == clean_title]

    if match.empty:
        return None
    
    book_idx = match.index[0]
    row_number = books_df.index.get_loc(book_idx)

    tfidf_vector = tfidf_matrix[row_number]
    distance, indices = nn_model.kneighbors(tfidf_vector, n_neighbors=n_neighbors+1)
    recommended_indices = indices.flatten()[1:]
    recs = books_df.iloc[recommended_indices][[
        'original_title', 'author', 'description', 'image_url', 'avg_rating', 'genre_list'
    ]].copy()

    recs['genre_list'] = recs['genre_list'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else str(x)
    )
    return recs.drop_duplicates()


# Streamlit UI
st.title(" Book Reccomendation App")

user_input = st.text_input("Enter a book title you like: ")

if user_input:
    recs = recommend(user_input, books_df, vectorizer, nn_model,tfidf_matrix)

    if recs is not None:
        st.subheader("Reccomndations: ")
        for _, row in recs.iterrows():
            st.markdown(f"**{row['original_title']}** by *{row['author']}*")
            st.write(f"⭐ Avg Rating: {row['avg_rating']}")
            st.write(f"📚 Genres: {row['genre_list']}")
            if isinstance(row['image_url'], str) and row['image_url'].startswith("http"):
                st.image(row['image_url'], width=150)
            st.write(f"📝 {row['description'][:300]}...")
            st.markdown("---")
    else:
        st.error("Book not found. Try a different title.")