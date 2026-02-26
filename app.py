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
    
    t1 = preprocess(text1)
    t2 = preprocess(text2)
    
    tfidf_1 = word_vectorizer.transform([t1])
    tfidf_2 = word_vectorizer.transform([t2])
    
    char_1 = char_vectorizer.transform([t1])
    char_2 = char_vectorizer.transform([t2])
    
    jaccard = jaccard_similarity(t1, t2)
    cosine_val = cosine_similarity(tfidf_1, tfidf_2)[0][0]
    length_diff = abs(len(t1) - len(t2))
    
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

st.title("Plagiarism Detection System")

text1 = st.text_area("Enter Original Text")
text2 = st.text_area("Enter Suspicious Text")

if st.button("Check Plagiarism"):
    
    if text1 and text2:
        result, similarity = predict_plagiarism(text1, text2)
        
        if result == 1:
            st.error("Plagiarized ❌")
        else:
            st.success("Not Plagiarized ✅")
        
        st.write(f"Similarity Score: {similarity*100:.2f}%")
    
    else:
        st.warning("Please enter both texts.")