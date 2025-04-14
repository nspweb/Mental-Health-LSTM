import streamlit as st
import tensorflow as tf
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download resource NLTK jika belum
nltk.download('stopwords')
nltk.download('wordnet')

# Load model (gunakan path relatif untuk deployment)
model = tf.keras.models.load_model('model_mental_health_v1.h5')

# Inisialisasi Tokenizer (harus sama dengan saat training)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

# Fungsi preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenisasi, stopwords, dan lemmatization
    words = text.split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(cleaned_words)

# Antarmuka Streamlit
st.title("Mental Health Prediction (LSTM)")
st.write("Aplikasi ini memprediksi apakah teks yang Anda masukkan berkaitan dengan **Anxiety** atau **Depression**.")

input_text = st.text_area("Masukkan teks di sini:")

if st.button("Prediksi"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        cleaned_text = preprocess_text(input_text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

        prediction = model.predict(padded_sequence)
        label_index = np.argmax(prediction)

        if label_index == 0:
            st.success("Hasil prediksi: **Anxiety**")
        else:
            st.success("Hasil prediksi: **Depression**")