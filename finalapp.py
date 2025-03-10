import streamlit as st
import PyPDF2
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Function to compute similarity score
def compute_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    docs = [resume_text, job_desc]
    tfidf_matrix = vectorizer.fit_transform(docs)
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return round(float(similarity_score[0][0]) * 100, 2)  # Convert to percentage

# Streamlit UI
st.title("AI-Powered Resume Screening System")
st.header("Match Your Resume With Job Description")

# Job Description Input
job_desc = st.text_area("Enter Job Description:")

# Resume Upload
uploaded_file = st.file_uploader("Upload Resume (PDF Format)", type=["pdf"])

if uploaded_file is not None and job_desc:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_text = preprocess_text(resume_text)
        job_desc = preprocess_text(job_desc)
        match_score = compute_similarity(resume_text, job_desc)
    
    st.success(f"Resume Match Score: {match_score}%")
    if match_score > 70:
        st.write("✅ Strong Match! Consider applying.")
    elif match_score > 50:
        st.write("⚠️ Moderate Match. Improve your resume.")
    else:
        st.write("❌ Weak Match. Consider tailoring your resume.")
