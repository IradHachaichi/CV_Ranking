from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(cv_text, jd_text):
    """
    Compute cosine similarity between CV text and job description text using SentenceTransformer.
    
    Args:
        cv_text (str): The CV text to compare.
        jd_text (str): The job description text to compare.
    
    Returns:
        float: Cosine similarity score between 0 and 1, or None if jd_text is empty.
    """
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Check if jd_text is empty
    if not jd_text.strip():
        print("No job description provided. Skipping similarity scoring.")
        return None
    
    # Encode texts and compute similarity
    cv_embedding = model.encode(cv_text)
    jd_embedding = model.encode(jd_text)
    score = cosine_similarity([cv_embedding], [jd_embedding])[0][0]
    
    return score