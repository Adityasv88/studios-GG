import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Download NLTK Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Text Cleaning Function
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Vectorization Function
def vectorize_text(text_data, fit=True, vectorizer=None):
    if fit:
        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(text_data).toarray()
        return X, tfidf
    else:
        X = vectorizer.transform(text_data).toarray()
        return X

# Model Training Function
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Streamlit Text Classification Page with File Upload and URL Input
def text_classification():
    st.title("Text Classification")

    # Option to choose between File Upload or URL Input
    st.subheader("Choose Dataset Input Method")
    input_method = st.radio("Select an option:", ('File Upload', 'URL Input'))

    data = None  # Initialize dataset as None

    # Step 1: Dataset Input Section
    if input_method == 'File Upload':
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
    else:  # URL Input
        dataset_url = st.text_input("Enter dataset URL")
        if dataset_url:
            try:
                data = pd.read_csv(dataset_url)
                st.success("Dataset loaded successfully")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    # If data is loaded successfully
    if data is not None:
        st.write("Dataset Preview")
        st.dataframe(data.head())

        # Check if the dataset contains 'description' and 'category' columns
        if 'description' in data.columns and 'category' in data.columns:
            
            # Step 2: Text Preprocessing
            st.subheader("Text Preprocessing")
            if st.button("Clean and Preprocess Text"):
                data['cleaned_text'] = data['description'].apply(clean_text)
                st.write("Cleaned Text Sample")
                st.dataframe(data[['description', 'cleaned_text']].head())
            
            # Step 3: Vectorize and Train Model
            if st.button("Train Model"):
                X, vectorizer = vectorize_text(data['cleaned_text'], fit=True)
                y = data['category']
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = train_model(X_train, y_train)
                st.success("Model Trained Successfully")
                
                # Step 4: Evaluate the model
                st.subheader("Model Evaluation")
                accuracy, report = evaluate_model(model, X_test, y_test)
                st.write(f"Accuracy: {accuracy}")
                st.text("Classification Report")
                st.text(report)
                
                # Save the model and vectorizer
                joblib.dump(model, 'ecommerce_text_classifier.pkl')
                joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
                st.success("Model and Vectorizer saved!")
                
    # Step 5: Future Predictions
    st.subheader("Classify New Text")
    
    model_file = 'ecommerce_text_classifier.pkl'
    vectorizer_file = 'tfidf_vectorizer.pkl'
    
    if st.button("Classify Text"):
        # Load saved model and vectorizer
        if model_file and vectorizer_file:
            model = joblib.load(model_file)
            vectorizer = joblib.load(vectorizer_file)
            
            new_text = st.text_input("Enter text for classification")
            
            if new_text:
                cleaned_new_text = clean_text(new_text)
                vectorized_new_text = vectorize_text([cleaned_new_text], fit=False, vectorizer=vectorizer)
                prediction = model.predict(vectorized_new_text)
                st.write(f"Predicted Category: {prediction[0]}")
