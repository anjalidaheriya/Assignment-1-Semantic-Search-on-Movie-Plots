import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset and create embeddings (global for testing)
df = None
model = None
embeddings = None

def load_data_and_create_embeddings():
    """
    Load the movies dataset and create embeddings using all-MiniLM-L6-v2 model.
    This function should be called before using search_movies.
    """
    global df, model, embeddings
    
    # Load the movies dataset
    df = pd.read_csv('movies.csv')
    
    # Load the Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert the plot of the movies into embeddings
    plot_texts = df['plot'].tolist()
    embeddings = model.encode(plot_texts, convert_to_tensor=False)
    
    print(f"Loaded {len(df)} movies and created embeddings with shape: {embeddings.shape}")

def search_movies(query, top_n=5):
    """
    Search for movies based on semantic similarity to the query.
    
    Args:
        query (str): The search query
        top_n (int): Number of top results to return (default: 5)
    
    Returns:
        pd.DataFrame: DataFrame with columns: title, plot, similarity
                     sorted by similarity score in descending order
    """
    global df, model, embeddings
    
    # Check if data and embeddings are loaded
    if df is None or embeddings is None:
        load_data_and_create_embeddings()
    
    # Encode the query
    query_embedding = model.encode([query], convert_to_tensor=False)
    
    # Calculate cosine similarity between query and all movie plots
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    
    # Create results DataFrame
    results = df.copy()
    results['similarity'] = similarities
    
    # Sort by similarity score (descending) and get top_n results
    results = results.sort_values('similarity', ascending=False).head(top_n)
    
    # Reset index for clean output
    results = results.reset_index(drop=True)
    
    return results

# Load data and create embeddings when module is imported
if __name__ == "__main__":
    load_data_and_create_embeddings()
    
    # Test the search functionality
    test_query = "spy thriller in Paris"
    print(f"\nTesting search with query: '{test_query}'")
    results = search_movies(test_query, top_n=3)
    print("\nTop 3 results:")
    print(results[['title', 'plot', 'similarity']])
else:
    # When imported as module, load data automatically
    load_data_and_create_embeddings()
