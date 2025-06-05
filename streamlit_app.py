import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import fitz  # PyMuPDF
import torch
import os

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Chunk the text
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Load QA model
@st.cache_resource
def load_qa_pipeline():
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

# Run QA
def answer_question(question, text_chunks, qa_pipeline):
    best_answer = {"score": -1, "answer": "No relevant answer found."}
    for chunk in text_chunks:
        try:
            result = qa_pipeline(question=question, context=chunk)
            if result["answer"].strip() and result["score"] > best_answer["score"]:
                best_answer = result
        except:
            continue
    return best_answer["answer"]

# Streamlit UI
st.set_page_config(page_title="PDF Q&A App", layout="centered")
st.title("📘 PDF Question Answering")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("🔍 Extracting text..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        text_chunks = chunk_text(pdf_text)

    st.success("✅ Text extracted. Ask your question below.")

    question = st.text_input("❓ Ask a question about the document:")

    if question:
        with st.spinner("🤖 Finding answer..."):
            qa_pipeline = load_qa_pipeline()
            answer = answer_question(question, text_chunks, qa_pipeline)
        st.markdown(f"📌 Answer: {answer}")
