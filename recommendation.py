import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("dataset.csv")

# Check if data is loaded correctly
print(df.head())  

# Ensure product_id exists before running the function
if "product_id" not in df.columns:
    raise ValueError("Column 'product_id' not found in dataset.csv!")

# Combine category and description for better representation
df["features"] = df["category"] + " " + df["description"]

# Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
feature_vectors = vectorizer.fit_transform(df["features"])

# Compute similarity matrix
similarity_matrix = cosine_similarity(feature_vectors)

def get_recommendations(product_id, top_n=3):
    if product_id not in df["product_id"].values:
        print(f"Product ID {product_id} not found!")
        return []
    
    idx = df[df["product_id"] == product_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_products = [df.iloc[i[0]]["product_id"] for i in similarity_scores[1:top_n+1]]
    
    return recommended_products

# Test function
if __name__ == "__main__":
    print(get_recommendations(101))