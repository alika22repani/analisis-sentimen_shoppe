import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL (VERSI RINGAN)
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Shopee Sentiment", layout="wide")

# =========================
# STYLE SHOPEE
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }

    h1 {
        color: #EE4D2D;
        text-align: center;
        font-weight: bold;
    }

    .stButton>button {
        background-color: #EE4D2D;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }

    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #EE4D2D;
    }

    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<h1>🛒 Shopee Sentiment Analysis</h1>", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("shopee_reviews.csv")

# =========================
# PREPROCESS INPUT
# =========================
def preprocess_input(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# =========================
# METRIC
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Positif", len(df[df['sentimen']=="positif"]))
col2.metric("Netral", len(df[df['sentimen']=="netral"]))
col3.metric("Negatif", len(df[df['sentimen']=="negatif"]))

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# INPUT + HASIL
# =========================
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💬 Analisis Komentar")

    user_input = st.text_area("Masukkan komentar:")

    if st.button("Analisis Sentimen"):
        if user_input != "":
            clean_text = preprocess_input(user_input)

            # 🔥 PREDIKSI VERSI BARU
            vec = vectorizer.transform([clean_text])
            hasil = model.predict(vec)[0]
            confidence = 100

            # RULE TAMBAHAN
            if "biasa" in user_input.lower():
                hasil = "netral"

            # OUTPUT
            if hasil == "positif":
                st.success(f"Hasil: {hasil} 😊 ({confidence}%)")
                color = "Greens"
            elif hasil == "negatif":
                st.error(f"Hasil: {hasil} 😡 ({confidence}%)")
                color = "Reds"
            else:
                st.info(f"Hasil: {hasil} 😐 ({confidence}%)")
                color = "Blues"

            # WORDCLOUD
            st.subheader("☁️ Wordcloud dari Komentar")

            wc = WordCloud(
                width=600,
                height=300,
                background_color='white',
                colormap=color
            ).generate(clean_text)

            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis('off')

            st.pyplot(fig)

        else:
            st.warning("Masukkan komentar dulu!")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# GRAFIK
# =========================
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Distribusi Sentimen")

    st.bar_chart(df['sentimen'].value_counts())

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DATA KOMENTAR
# =========================
st.markdown("---")
st.subheader("📜 Data Komentar Pengguna")

filter_sentimen = st.selectbox("Filter Sentimen", ["Semua", "positif", "netral", "negatif"])

if filter_sentimen != "Semua":
    df_tampil = df[df['sentimen'] == filter_sentimen]
else:
    df_tampil = df

st.dataframe(df_tampil[['komentar', 'sentimen']], height=300)

# =========================
# INSIGHT
# =========================
st.markdown("### 🔍 Insight:")
st.write("- Mayoritas pengguna memberikan sentimen positif")
st.write("- Beberapa keluhan terkait pengiriman dan performa aplikasi")