from tqdm import tqdm
from itertools import islice
from youtube_comment_downloader import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import matplotlib.pyplot as plt
import csv
import streamlit as st
import pandas as pd
import base64

st.title("Youtube Comment Sentimen Analisis")
st.write("Program ini akan menganalisis komentar dalam sebuah video di youtube menggunakan sentiment analysis, tidak termasuk komentar dalam komentar.")

# Input untuk memilih model
selected_model = st.selectbox("Pilih Model Sentiment Analisis", ["Mdhugol-Indonesia", "Nlptown-Multilingual", "Hw2942-Chinese"])

# Membuat dictionary untuk menentukan model dan label index sesuai pilihan
model_configs = {
    "Mdhugol-Indonesia": {
        "pretrained": "mdhugol/indonesia-bert-sentiment-classification",
        "label_index": {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
    },
    "Nlptown-Multilingual": {
        "pretrained": "nlptown/bert-base-multilingual-uncased-sentiment",
        "label_index": {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
    },
    "Hw2942-Chinese": {
        "pretrained": "hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2",
        "label_index": {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
    }
}

# Menggunakan model yang dipilih
selected_config = model_configs[selected_model]

# Inisialisasi model dan tokenizer
pretrained = selected_config["pretrained"]
model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}


# Input URL video
video_url = st.text_input("Masukkan URL video YouTube:")

# Input jumlah komentar yang ingin diambil
num_comments = st.number_input("Jumlah komentar yang ingin diambil:", min_value=1, value=10)

# Fungsi untuk analisis sentimen
def analisis_sentimen(text):
    result = sentiment_analysis(text)
    label = label_index[result[0]['label']]
    score = result[0]['score'] * 100
    return label, score

if st.button("Mulai Analisis"):
    
    #Memulai Download Komentar
    st.info("Memulai Download Komentar....")
    # Inisialisasi YoutubeCommentDownloader
    downloader = YoutubeCommentDownloader()

    # Mendapatkan komentar
    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)

    # Membuka file CSV untuk menulis
    with open('comments.csv', mode='w', encoding='utf-8', newline='') as file:
        # Membuat objek writer
        writer = csv.DictWriter(file, fieldnames=['cid', 'text', 'time', 'author', 'channel', 'votes', 'photo', 'heart', 'reply'])
        
        # Menulis header
        writer.writeheader()
        
        # Menulis data komentar
        for comment in tqdm(islice(comments, num_comments)):
            # Menghapus kolom 'time_parsed' dari komentar
            comment.pop('time_parsed', None)
            writer.writerow(comment)

    st.success(f"Komentar berhasil diunduh dan disimpan dalam file 'comments.csv'")
    
    # Membaca data dari file CSV
    comments_df = pd.read_csv('comments.csv')
    
    #analisis sentimen
    st.info("Memulai analisis sentimen....")

    # List untuk menyimpan hasil analisis sentimen
    # List untuk menyimpan skor sentimen
    scores = []
    labels = []

    # Membaca data dari file CSV
    with open('comments.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader):
            comment_text = row['text']
             # Bagi teks menjadi bagian-bagian dengan panjang maksimum 512 token
            parts = [comment_text[i:i+512] for i in range(0, len(comment_text), 512)]
            for part in parts:
                # Analisis sentimen
                result = sentiment_analysis(part)
                label = label_index[result[0]['label']]            
                score = result[0]['score'] * 100
                labels.append(label)
                scores.append(score)
                #hasil_analisis.append((comment_text, label, score))

    # Menampilkan hasil analisis sentimen
    st.subheader("Hasil Analisis Sentimen")
    #st.write(hasil_analisis)

    # Menampilkan histogram
    #labels, scores = zip(*[(label, score) for _, label, score in hasil_analisis])
    plt.hist(labels, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Skor Sentimen')
    plt.ylabel('Jumlah Komentar')
    plt.title('Distribusi Sentimen Komentar')
    st.pyplot(plt)

    # Menghitung jumlah dan persentase
    jumlah_positif = labels.count('positive')
    jumlah_negatif = labels.count('negative')
    jumlah_netral = labels.count('neutral')
    total_komentar = len(labels)
    persentase_positif = (jumlah_positif / total_komentar) * 100
    persentase_negatif = (jumlah_negatif / total_komentar) * 100
    persentase_netral = (jumlah_netral / total_komentar) * 100

    st.write(f"Total Komentar: {total_komentar}")
    st.write(f"Persentase Komentar Positif: {persentase_positif:.2f}% / {jumlah_positif} Komentar")
    st.write(f"Persentase Komentar Negatif: {persentase_negatif:.2f}% / {jumlah_negatif} Komentar")
    st.write(f"Persentase Komentar Netral: {persentase_netral:.2f}% / {jumlah_netral} Komentar")
   
    # Menambahkan teks dengan ukuran kecil
    st.markdown("<p style='font-size:small;'>Komentar yang lebih panjang dari 512 karakter akan dibagi menjadi dua, sehingga total komentar mungkin lebih dari yang Anda ambil.</p>", unsafe_allow_html=True)
    # Menampilkan tabel dengan menggunakan st.table()
    #st.subheader("Data Komentar")
    #st.table(comments_df)
    
   

