import streamlit as st
import re
import string
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# === Sidebar ===
st.sidebar.title("Mental Health Sentiment Analyzer")
st.sidebar.write("Enter your text to predict the sentiment!")

# === Load Tokenizer ===
@st.cache_resource
def load_tokenizer(path='D:\3 matkul\Mental Health Using LSTM\src\tokenizer.pickle'):
    try:
        if not os.path.exists(path):
            st.error(f"❌ File tokenizer tidak ditemukan di path: {path}.")
            return None
        with open(path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memuat tokenizer: {e}")
        return None

# === Load Trained Model ===
@st.cache_resource
def load_trained_model(model_path='D:\3 matkul\Mental Health Using LSTM\src\model_mental_health_v1.keras'):
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ File model tidak ditemukan di path: {model_path}.")
            return None
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# === Text Cleaning Function ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)  # remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # remove words containing numbers
    return text.strip()

# === Predict Function ===
def predict_sentiment(model, tokenizer, text, maxlen=100):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded)
    labels = ['Normal', 'Stress', 'Depression']  # Replace with your actual labels
    predicted_label = labels[prediction.argmax()]
    return predicted_label, prediction.max() * 100

# === Main App ===
def main():
    # Display title and description
    st.title("🌱 Mental Health Text Analysis")
    st.markdown("""
    This tool predicts whether the text you input relates to **Anxiety** or **Depression**.
    """)
    
    # Text input
    input_text = st.text_area("Enter your text here:", height=200)
    
    if st.button("Analyze Text"):
        if not input_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            # Load model and tokenizer
            tokenizer = load_tokenizer()
            model = load_trained_model()
            
            if model and tokenizer:
                st.success("✅ Model and tokenizer loaded!")

                # Predict sentiment
                predicted_label, confidence = predict_sentiment(model, tokenizer, input_text)
                
                # Display the result
                if predicted_label == 'Normal':
                    st.markdown(f"""
                    <div style='background-color:#E8E8FF; padding:15px; border-radius:10px;'>
                        <h3 style='margin:0; color:#00008B;'>Normal Sentiment Detected</h3>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif predicted_label == 'Stress':
                    st.markdown(f"""
                    <div style='background-color:#FFE2E2; padding:15px; border-radius:10px;'>
                        <h3 style='margin:0; color:#D8000C;'>Stress Detected</h3>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color:#F3F3F3; padding:15px; border-radius:10px;'>
                        <h3 style='margin:0; color:#8B0000;'>Depression Detected</h3>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Add disclaimer
                st.markdown("""
                <div style='font-size:0.8em; margin-top:20px;'>
                <b>Disclaimer:</b> This tool provides only a computational analysis and is not a substitute for professional mental health evaluation.
                If you or someone you know is struggling with mental health issues, please consult with a qualified healthcare professional.
                </div>
                """, unsafe_allow_html=True)

# === Run ===
if __name__ == '__main__':
    main()
