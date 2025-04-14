import streamlit as st
import tensorflow as tf
import numpy as np
import re
import string
import nltk
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Paths to model and tokenizer files
model_path = os.path.join('src', 'model_mental_health_v1.keras')
tokenizer_path = os.path.join('src', 'tokenizer.pickle')

# Define model architecture (identical to training model)
def create_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128),
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
        Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0003),
        metrics=['accuracy']
    )
    
    return model

# Try to load the model
try:
    # First try to load the saved model directly
    model = tf.keras.models.load_model(model_path)
    st.sidebar.success("Model loaded successfully")
except Exception as e:
    st.sidebar.warning(f"Could not load model directly. Attempting to reconstruct...")
    
    try:
        # Create model with the same architecture
        model = create_model()
        
        # Initialize with dummy input to build the model
        dummy_input = tf.zeros((1, 100))
        model(dummy_input)
        
        # Try to load weights if available
        model_weights_path = os.path.join('src', 'model_weights.h5')
        if os.path.exists(model_weights_path):
            model.load_weights(model_weights_path)
            st.sidebar.success("Model reconstructed and weights loaded successfully")
        else:
            st.sidebar.warning("Using model with random weights - predictions will not be accurate")
    except Exception as e:
        st.sidebar.error(f"Error creating model: {str(e)}")
        st.stop()

# Try to load the tokenizer
try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    st.sidebar.success("Tokenizer loaded successfully")
except Exception as e:
    st.sidebar.warning("Could not load tokenizer. Creating a simple tokenizer...")
    
    # Create a simple tokenizer as fallback
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    
    # Fit with some example texts
    sample_texts = [
        "i feel anxious about my future",
        "feeling depressed lately",
        "anxiety attacks are happening more frequently",
        "depression symptoms getting worse",
        "worried about everything",
        "feeling sad and hopeless",
        "cant sleep because of anxiety",
        "no motivation due to depression"
    ]
    tokenizer.fit_on_texts(sample_texts)
    st.sidebar.info("Using simple tokenizer - predictions may not be accurate")

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and mentions
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # Remove non-alphabetic characters and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

# Streamlit UI
st.markdown("""
    <style>
        .main {
            background-color: #F8F9FA;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .css-1v3fvcr {
            background-color: #DFF6FF;
            padding: 10px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Mental Health Text Analysis")
st.markdown("""
This application predicts whether the text you enter relates to **Anxiety** or **Depression**.
""")

col1, col2 = st.columns(2)
with col1:
    input_text = st.text_area("Enter your text here:", height=200)
with col2:
    try:
        st.image("src/images.jpg", use_column_width=True)
    except:
        st.info("Image not found. Place an image named 'images.jpg' in the src folder for visual enhancement.")

if st.button("Analyze Text", key="predict"):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        # Preprocess the input text
        cleaned_text = preprocess_text(input_text)
        
        # Debug information (can be removed in production)
        with st.expander("View preprocessing details"):
            st.write(f"**Original text:** {input_text}")
            st.write(f"**Preprocessed text:** {cleaned_text}")
        
        # Convert to sequence and pad
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        
        # Predict
        with st.spinner("Analyzing..."):
            try:
                prediction = model.predict(padded_sequence)
                label_index = np.argmax(prediction)
                confidence = prediction[0][label_index] * 100
                
                # Display results
                st.markdown("### Analysis Result")
                
                if label_index == 0:
                    st.markdown(f"""
                    <div style='background-color:#FFE2E2; padding:15px; border-radius:10px;'>
                        <h3 style='margin:0; color:#D8000C;'>Anxiety Detected</h3>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color:#E8E8FF; padding:15px; border-radius:10px;'>
                        <h3 style='margin:0; color:#00008B;'>Depression Detected</h3>
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
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Add information in the sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application uses a deep learning model with LSTM architecture to analyze text and identify patterns 
associated with anxiety or depression. It's intended for educational purposes only.
""")

# Add resources
st.sidebar.header("Mental Health Resources")
st.sidebar.markdown("""
* [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/): 1-800-273-8255
* [Crisis Text Line](https://www.crisistextline.org/): Text HOME to 741741
* [SAMHSA Treatment Referral Hotline](https://www.samhsa.gov/find-help/national-helpline): 1-800-662-4357
""")