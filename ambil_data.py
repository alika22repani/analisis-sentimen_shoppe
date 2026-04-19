from google_play_scraper import reviews, Sort
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =========================
# AMBIL DATA
# =========================
result, _ = reviews(
    'com.shopee.id',
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=2000
)

df = pd.DataFrame(result)
df = df[['content', 'score', 'at']]
df.columns = ['komentar', 'rating', 'timestamp']

print("Data berhasil diambil!")
print(df.head())

# =========================
# LABEL SENTIMEN
# =========================
def label_sentimen(rating):
    if rating >= 4:
        return 'positif'
    elif rating == 3:
        return 'netral'
    else:
        return 'negatif'

df['sentimen'] = df['rating'].apply(label_sentimen)
print("Sentimen berhasil ditambahkan!")

# =========================
# PREPROCESSING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['clean'] = df['komentar'].apply(clean_text)
print("Preprocessing selesai!")

# =========================
# TOKENIZING
# =========================
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['clean'])

X = tokenizer.texts_to_sequences(df['clean'])
X = pad_sequences(X, maxlen=100)

print("Tokenizing selesai!")

# =========================
# MODEL LSTM
# =========================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['sentimen'])

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64))
model.add(LSTM(64))
model.add(Dense(3, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X, y, epochs=3, batch_size=32)

print("Model selesai dilatih!")

# =========================
# VISUALISASI GRAFIK
# =========================
df['sentimen'].value_counts().plot(kind='bar')
plt.title("Distribusi Sentimen Shopee")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah")

plt.show(block=False)
plt.pause(2)
plt.close()

# =========================
# WORDCLOUD SEMUA DATA
# =========================
text = " ".join(df['clean'].astype(str))

wc = WordCloud(width=800, height=400).generate(text)

plt.figure()
plt.imshow(wc)
plt.axis('off')
plt.title("Wordcloud Shopee")

plt.show(block=False)
plt.pause(2)
plt.close()

# =========================
# WORDCLOUD PER SENTIMEN
# =========================
df_pos = df[df['sentimen'] == 'positif']
df_neg = df[df['sentimen'] == 'negatif']
df_net = df[df['sentimen'] == 'netral']

text_pos = " ".join(df_pos['clean'].astype(str))
text_neg = " ".join(df_neg['clean'].astype(str))
text_net = " ".join(df_net['clean'].astype(str))

wc_pos = WordCloud(width=800, height=400, colormap='Greens').generate(text_pos)
wc_neg = WordCloud(width=800, height=400, colormap='Reds').generate(text_neg)
wc_net = WordCloud(width=800, height=400, colormap='Blues').generate(text_net)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(wc_pos)
plt.axis('off')
plt.title("Positif")

plt.subplot(1,3,2)
plt.imshow(wc_net)
plt.axis('off')
plt.title("Netral")

plt.subplot(1,3,3)
plt.imshow(wc_neg)
plt.axis('off')
plt.title("Negatif")

plt.show(block=False)
plt.pause(2)
plt.close()

# =========================
# SIMPAN DATA
# =========================
df.to_csv('shopee_reviews.csv', index=False)
print("Data CSV disimpan!")

# =========================
# SIMPAN MODEL
# =========================
print("MASUK BAGIAN SIMPAN")

model.save("model.h5")

import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model & tokenizer berhasil disimpan!")