import streamlit as st
import openai
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import re
from fpdf import FPDF
from io import BytesIO

# --- Configuration ---
st.set_page_config(page_title="AI Resume Analyzer & Builder", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


# --- Helper Functions ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return re.sub(r'\s+', ' ', text).strip()


def get_similarity_score(resume_text, job_description):
    embeddings = sbert_model.encode([resume_text, job_description], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity), 2)


def gpt_feedback(resume_text, job_description=""):
    prompt = f"""
You are a career coach. Analyze this resume and provide:
1. Resume Score (Good, Average, Poor)
2. Strengths
3. Weaknesses
4. Suggestions
5. Improved summary or bullet points

Resume:
\"\"\"{resume_text}\"\"\"

Job Description (if any):
\"\"\"{job_description}\"\"\"
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content']


def gpt_generate_resume(name, title, summary, skills, experience, education):
    prompt = f"""
Create a professional resume with the following:
- Name: {name}
- Title: {title}
- Summary: {summary}
- Skills: {skills}
- Experience: {experience}
- Education: {education}

Format it in clear sections with bullet points and professional language.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content']


def export_to_pdf(text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    lines = text.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, line)
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer


# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Analyze Resume", "🛠️ Build Resume"])

# --- Analyze Resume Tab ---
with tab1:
    st.header("📊 AI Resume Analyzer")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_desc = st.text_area("Optional: Paste Job Description", height=200)

    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.subheader("📄 Extracted Resume Text")
        st.text_area("Resume Text", resume_text, height=300)

        if st.button("🔍 Analyze Resume"):
            with st.spinner("Analyzing with GPT-4..."):
                feedback = gpt_feedback(resume_text, job_desc)
                similarity = get_similarity_score(resume_text, job_desc) if job_desc else "N/A"
            st.success("Analysis Complete")
            st.subheader("🧠 Feedback from GPT-4")
            st.markdown(feedback)
            if job_desc:
                st.subheader("🔗 Job Match Score")
                st.write(f"**Similarity Score:** {similarity} (0 to 1)")

# --- Resume Builder Tab ---
with tab2:
    st.header("🛠️ AI Resume Builder")
    name = st.text_input("Full Name")
    title = st.text_input("Professional Title (e.g., Software Engineer)")
    summary = st.text_area("Summary")
    skills = st.text_area("Skills (comma-separated)")
    experience = st.text_area("Work Experience")
    education = st.text_area("Education")

    if st.button("⚙️ Generate Resume"):
        if name and title and summary and skills:
            with st.spinner("Generating resume with GPT-4..."):
                resume_output = gpt_generate_resume(name, title, summary, skills, experience, education)
            st.success("Resume Generated")
            st.subheader("📄 Resume Preview")
            st.text_area("Generated Resume", resume_output, height=400)

            pdf_buffer = export_to_pdf(resume_output)
            st.download_button("📤 Download as PDF", data=pdf_buffer, file_name="resume.pdf", mime="application/pdf")
        else:
            st.warning("Please fill at least Name, Title, Summary, and Skills.")

