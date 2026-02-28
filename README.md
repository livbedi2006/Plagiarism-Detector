# 🧠 Plagiarism Detection System
 
A sophisticated plagiarism detection application that uses hybrid NLP techniques combined with TF-IDF vectorization to identify content similarity between texts.
 
## 🌟 Features
 
- **Hybrid Approach**: Combines multiple similarity metrics for accurate detection
- **Real-time Analysis**: Instant similarity scoring with detailed metrics
- **Interactive UI**: Beautiful Streamlit interface with color-coded results
- **Multiple Similarity Metrics**: 
  - Cosine Similarity (TF-IDF based)
  - Jaccard Similarity (word overlap)
  - Length difference analysis
- **Risk Assessment**: Visual indicators for plagiarism risk levels
 
## 🛠️ Technology Stack
 
- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn
- **NLP**: NLTK (stopwords, lemmatization)
- **Vectorization**: TF-IDF (word-level and character-level)
- **Algorithm**: Logistic Regression with hybrid features
 
## 📊 How It Works
 
1. **Text Preprocessing**: 
   - Lowercase conversion
   - Punctuation removal
   - Stopword elimination
   - Word lemmatization
 
2. **Feature Extraction**:
   - Word-level TF-IDF vectors (1-3 grams)
   - Character-level TF-IDF vectors (3-5 grams)
   - Jaccard similarity score
   - Cosine similarity score
   - Text length difference
 
3. **Classification**:
   - Logistic Regression model trained on hybrid features
   - Binary classification (plagiarized vs. original)
 
4. **Similarity Scoring**:
   - Percentage-based similarity display
   - Color-coded risk levels:
     - 🔴 Red: 80%+ (High Risk)
     - 🟡 Yellow: 50-79% (Moderate)
     - 🟢 Green: <50% (Low Risk)
 
## 🚀 Installation
 
1. Clone the repository:
```bash
git clone https://github.com/yourusername/plagiarism-detector.git
cd plagiarism-detector/plagiarism_app
```
 
2. Install dependencies:
```bash
pip install -r requirements.txt
```
 
3. Run the application:
```bash
streamlit run app.py
```
 
## 📁 Project Structure
 
```
plagiarism_detector/
├── plagiarism_app/
│   ├── app.py                 # Main Streamlit application
│   ├── requirements.txt       # Python dependencies
│   ├── plagiarism_model.pkl   # Trained ML model
│   ├── word_vectorizer.pkl    # Word-level TF-IDF vectorizer
│   └── char_vectorizer.pkl    # Character-level TF-IDF vectorizer
├── nlp.ipynb                 # Training notebook
├── Dataset.csv               # Training dataset
└── README.md                 # This file
```
 
## 🎯 Usage
 
1. Open the application in your browser (usually http://localhost:8501/)
2. Enter the original text in the first text area
3. Enter the suspicious text in the second text area
4. Click "Check Plagiarism" to analyze
5. View the results:
   - Plagiarism detection status
   - Similarity percentage with color coding
   - Detailed similarity metrics
   - Risk assessment
 
## 📈 Model Performance
 
The model is trained on a hybrid feature set combining:
- **Word-level TF-IDF**: Captures semantic similarity
- **Character-level TF-IDF**: Detects structural patterns
- **Jaccard Similarity**: Measures word overlap
- **Cosine Similarity**: Vector space similarity
- **Length Features**: Identifies unusual length variations
 
## 🔧 Model Training
 
The model is trained using the `nlp.ipynb` notebook which includes:
- Data preprocessing and cleaning
- Feature engineering
- Model training with Logistic Regression
- Performance evaluation
- Model serialization
 
## 🎨 UI Features
 
- **Modern Design**: Gradient background with rounded components
- **Responsive Layout**: Centered layout for optimal viewing
- **Visual Feedback**: Progress bars and color-coded results
- **Detailed Metrics**: Multiple similarity scores for comprehensive analysis
 
## 📊 Similarity Metrics Explained
 
- **Cosine Similarity**: Measures the cosine of the angle between two vectors (0-1)
- **Jaccard Similarity**: Ratio of intersection to union of word sets (0-1)
- **Percentage Display**: Human-readable similarity percentage
 
## 🚨 Limitations
 
- Works best with English text
- Requires sufficient text length for accurate analysis
- May need additional training for domain-specific content
- Performance depends on quality of training data
 
## 🤝 Contributing
 
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
 
 
## 👨‍💻 Author
 
Created by [Livjot singh] - [www.livjotseerat@gmail.com]
 
## 🙏 Acknowledgments
 
- NLTK for natural language processing tools
- Scikit-learn for machine learning algorithms
- Streamlit for the web application framework
 
