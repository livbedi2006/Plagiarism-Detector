import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved model and vectorizers
model = pickle.load(open("plagiarism_model.pkl", "rb"))
word_vectorizer = pickle.load(open("word_vectorizer.pkl", "rb"))
char_vectorizer = pickle.load(open("char_vectorizer.pkl", "rb"))

# Download nltk resources (first time only)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    if len(set1 | set2) == 0:
        return 0
    return len(set1 & set2) / len(set1 | set2)

def predict_plagiarism(text1, text2):
tfidf_1 = word_vectorizer.transform([text1])
tfidf_2 = word_vectorizer.transform([text2])
    
    char_1 = char_vectorizer.transform([text1])
    char_2 = char_vectorizer.transform([text2])
    
    jaccard = jaccard_similarity(text1, text2)
    cosine_val = cosine_similarity(tfidf_1, tfidf_2)[0][0]
    length_diff = abs(len(text1) - len(text2))
    
    extra = np.array([[jaccard, cosine_val, length_diff]])
    
    final_input = hstack([
        tfidf_1,
        tfidf_2,
        char_1,
        char_2,
        extra
    ])
    
    prediction = model.predict(final_input)[0]
    return prediction, cosine_val

# ------------------ Streamlit UI ------------------

st.markdown("<h1 style='text-align: center; color: white;'>🧠 Plagiarism Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: lightgray;'>Hybrid NLP + TF-IDF Based Detection</p>", unsafe_allow_html=True)

text1 = st.text_area("Enter Original Text")
text2 = st.text_area("Enter Suspicious Text")

if st.button("Check Plagiarism"):
    
    if text1 and text2:
        result, similarity = predict_plagiarism(text1, text2)

        st.markdown("---")

        if result == 1:
            st.error("🚨 Plagiarized Content Detected!")
        else:
            st.success("✅ Content Appears Original")

        st.write("### Similarity Score")

        percentage = int(similarity * 100)

        st.progress(percentage)
        st.write(f"**{percentage}% Similarity**")
    
    else:
        st.warning("Please enter both texts.")
        st.set_page_config(
    page_title="Plagiarism Detection System",
    page_icon="🧠",
    layout="centered"
)
# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)