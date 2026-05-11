import pandas as pd
from sklearn.cluster import KMeans
import pickle

# Load dataset
file_path = "Mall_Customers.csv"
data = pd.read_csv(file_path)

# Select Features
X = data.iloc[:, [3, 4]].values

# Train KMeans
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

kmeans.fit(X)

# Save Model
pickle.dump(kmeans, open("model.pkl", "wb"))

print("Model trained successfully!")