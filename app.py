from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get uploaded file
    file = request.files["file"]

    # Get number of clusters
    clusters = int(request.form["clusters"])

    # Save uploaded file
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Read CSV file
    data = pd.read_csv(filepath)

    # Select features
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Create KMeans model
    kmeans = KMeans(
        n_clusters=clusters,
        random_state=42
    )

    # Predict clusters
    data["Cluster"] = kmeans.fit_predict(X)

    # ======================================
    # Scatter Graph Visualization
    # ======================================

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        x=data['Annual Income (k$)'],
        y=data['Spending Score (1-100)'],
        hue=data['Cluster'],
        palette='Set1',
        s=100
    )

    plt.title("Customer Segmentation Graph")

    plt.xlabel("Annual Income")

    plt.ylabel("Spending Score")

    plt.savefig("static/cluster_graph.png")

    plt.close()

    # ======================================
    # Pie Chart
    # ======================================

    cluster_counts = data["Cluster"].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(
        cluster_counts,
        labels=cluster_counts.index,
        autopct='%1.1f%%'
    )

    plt.title("Cluster Distribution")

    plt.savefig("static/pie_chart.png")

    plt.close()

    # ======================================
    # HTML Table
    # ======================================

    table = data.head(20).to_html(
        classes="table table-bordered table-striped",
        index=False
    )

    return render_template(
        "index.html",
        table=table,
        message=f"Segmentation completed with {clusters} clusters!"
    )


if __name__ == "__main__":
    app.run(debug=True)