import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to calculate similarity score
def calculate_similarity(job_desc, resume_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_desc, resume_text])
    similarity = cosine_similarity(tfidf_matrix)[0, 1]
    return round(similarity * 100, 2)  # Convert to percentage

# Streamlit UI
st.title("AI-Powered Resume Screening System")
st.subheader("Upload your Resume and Compare with Job Description")

# Input Job Description
job_desc = st.text_area("Enter Job Description:", height=200)

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

if uploaded_file and job_desc:
    resume_text = extract_text_from_pdf(uploaded_file)
    
    if resume_text:
        similarity_score = calculate_similarity(job_desc, resume_text)
        st.success(f"Resume Match Score: **{similarity_score}%**")
        
        if similarity_score > 75:
            st.balloons()
            st.write("✅ **Great Match!** Your resume aligns well with the job description.")
        elif similarity_score > 50:
            st.write("⚠️ **Moderate Match** - Consider improving your resume.")
        else:
            st.write("❌ **Weak Match** - You may need to revise your resume.")
    else:
        st.error("Could not extract text from the PDF. Try another file.")
