import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# load data
df = pd.read_csv("shopee_reviews.csv")

# fitur & label
X = df["komentar"]
y = df["sentimen"]

# ubah teks jadi angka
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

# model
model = LogisticRegression()
model.fit(X_train, y_train)

# simpan model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model berhasil dibuat!")