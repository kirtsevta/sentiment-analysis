import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Step 1: Load dataset
df = pd.read_csv("sentiment_dataset_1000.csv")  # make sure this file is in the same directory

# Step 2: Convert sentiment labels to numeric values
label_map = {
    "positive": 1,
    "neutral": 2,
    "negative": 0
}
df["label"] = df["sentiment"].map(label_map)

# Step 3: Separate features and target
X = df["text"]
y = df["label"]

# Step 4: Text vectorization using CountVectorizer
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Step 5: Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_vec, y)

# Step 6: Save the trained model and vectorizer
joblib.dump(model, "modeltest.pkl")
joblib.dump(vectorizer, "vectorizertest.pkl")

print("✅ Model and Vectorizer have been saved successfully!")
