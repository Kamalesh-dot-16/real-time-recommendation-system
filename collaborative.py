import pandas as pd
from surprise import Dataset, Reader, KNNBasic

# Load dataset
df = pd.read_csv("ratings.csv")

# Define reader format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# Train collaborative filtering model
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': True}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

def get_user_recommendations(user_id, top_n=3):
    products = df["product_id"].unique()
    predictions = []
    
    for product in products:
        pred = model.predict(user_id, product)
        predictions.append((product, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in predictions[:top_n]]

# Test function
if __name__ == "__main__":
    print(get_user_recommendations(1))