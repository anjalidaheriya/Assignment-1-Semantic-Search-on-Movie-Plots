# 🎬 Movie Semantic Search

This repository contains my implementation of a **semantic search engine for movie plots**.  
The goal of this assignment was to go beyond keyword search and instead use embeddings to capture the *meaning* of queries.

---

## 🔎 Project Overview

The system is built with **SentenceTransformers** (`all-MiniLM-L6-v2`) to encode movie plot descriptions into dense vector embeddings.  
When a user enters a query, the model computes its embedding and retrieves the most semantically similar movies using **cosine similarity**.

---

## ✨ Features

- **Semantic Understanding** – Queries are matched on meaning, not just keywords.  
- **Cosine Similarity Scoring** – Ranks results between `0` and `1` for easy interpretation.  
- **Configurable Results** – Adjustable `top_n` parameter to return as many results as needed.  
- **Pre-computed Embeddings** – All movie plots are embedded on initialization for fast retrieval.  


 ## Example Query

Test the function with the required query:
```python
search_movies('spy thriller in Paris')
```

**Expected Results**:
1. **Spy Movie** - "A spy navigates intrigue in Paris to stop a terrorist plot." (Similarity: ~0.77)
2. **Romance in Paris** - "A couple falls in love in Paris under romantic circumstances." (Similarity: ~0.39)
3. **Action Flick** - "A high-octane chase through New York with explosions." (Similarity: ~0.26)
## Dependencies

- sentence-transformers==5.1.0
- pandas
- scikit-learn
- numpy

## License

This project is part of the AI Systems Development assignment at IIIT Naya Raipur.
