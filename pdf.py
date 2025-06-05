import streamlit as st

import cohere

import PyPDF2

import base64

import os

import time

# === Secure API key from secrets ===

co = cohere.Client(st.secrets["cohere"]["api_key"])

def extract_text_from_pdf(file_path):

    pdf_reader = PyPDF2.PdfReader(file_path)

    text = ""

    for page in pdf_reader.pages:

        text += page.extract_text() or ""

    return text

def chunk_text(text, max_chunk_size=2500):

    chunks = []

    start = 0

    while start < len(text):

        end = start + max_chunk_size

        period_pos = text.rfind('.', start, end)

        if period_pos != -1:

            end = period_pos + 1

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start = end

    return chunks

def cohere_chat_summary(text):

    try:

        response = co.chat(

            model="command-xlarge-nightly",

            message=f"Summarize this text clearly and concisely:\n\n{text}",

            temperature=0.4,

            max_tokens=300

        )

        return response.text.strip()

    except Exception as e:

        return f"❌ API Error: {str(e)}"

def summarize_text(text):

    chunks = chunk_text(text)

    summaries = []

    for chunk in chunks:

        summary = cohere_chat_summary(chunk)

        summaries.append(summary)

        time.sleep(6)  # to respect API rate limits

    combined = " ".join(summaries)

    if len(combined) > 2000:

        return cohere_chat_summary(combined[:2000])

    return combined

def generate_auto_qa(text, num_questions=5):

    prompt = (

        f"Generate {num_questions} questions and answers from the following text:\n\n{text}\n\nFormat: Q1: ... A1: ... Q2: ... A2: ..."

    )

    try:

        response = co.chat(

            model="command-xlarge-nightly",

            message=prompt,

            temperature=0.5,

            max_tokens=600

        )

        return response.text.strip()

    except Exception as e:

        return f"❌ API Error: {str(e)}"

def generate_answer(text, question):

    prompt = f"Answer the question based on this text:\n\n{text}\n\nQuestion: {question}"

    try:

        response = co.chat(

            model="command-xlarge-nightly",

            message=prompt,

            temperature=0.3,

            max_tokens=200

        )

        return response.text.strip()

    except Exception as e:

        return f"❌ API Error: {str(e)}"

@st.cache_data

def display_pdf(file_path):

    with open(file_path, "rb") as f:

        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    pdf_view = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'

    st.markdown(pdf_view, unsafe_allow_html=True)

def get_download_link(content, filename, file_format):

    if file_format == "txt" or file_format == "doc":

        mime = "text/plain" if file_format == "txt" else "application/msword"

        data = content

    elif file_format == "csv":

        lines = content.split("\n")

        # Try formatting Q&A to CSV if possible

        if "Q1:" in content or "A1:" in content:

            rows = []

            question = ""

            answer = ""

            for line in lines:

                if line.strip().startswith("Q") and ":" in line:

                    question = line.split(":",1)[1].strip()

                elif line.strip().startswith("A") and ":" in line:

                    answer = line.split(":",1)[1].strip()

                    rows.append(f'"{question}","{answer}"')

            data = "\n".join(["Question,Answer"] + rows)

        else:

            # Just put text content in one csv column

            data = f"Text\n\"{content}\""

        mime = "text/csv"

    else:

        return None

    b64 = base64.b64encode(data.encode()).decode()

    href = f"data:{mime};base64,{b64}"

    return href

st.set_page_config(layout="wide")

def reset_outputs():

    """Clear all output-related session states."""

    st.session_state.output = ""

    st.session_state.output_type = ""

    st.session_state.show_download_options = False

    st.session_state.selected_format = None

    st.session_state.last_option = None

    st.session_state.last_qa_mode = None

def main():

    st.title("📄 PDF Summarizer + Q&A Tool")

    # Initialize session state variables if they don't exist

    for key in ["output", "output_type", "show_download_options", "selected_format", "last_option", "last_qa_mode"]:

        if key not in st.session_state:

            if key in ["show_download_options"]:

                st.session_state[key] = False

            else:

                st.session_state[key] = None if key in ["last_option", "last_qa_mode"] else ""

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # If uploaded_file changes, reset outputs immediately

    if "prev_uploaded_file_name" not in st.session_state:

        st.session_state.prev_uploaded_file_name = None

    # If user clears the uploader or uploads a different file, reset outputs

    if uploaded_file is None:

        # User cleared the uploader

        if st.session_state.prev_uploaded_file_name is not None:

            reset_outputs()

        st.session_state.prev_uploaded_file_name = None

        st.info("Please upload a PDF to start.")

        return

    else:

        if st.session_state.prev_uploaded_file_name != uploaded_file.name:

            reset_outputs()

            st.session_state.prev_uploaded_file_name = uploaded_file.name

    if uploaded_file.size > 10 * 1024 * 1024:

        st.error("❌ File too large! Please upload a PDF under 10MB.")

        return

    os.makedirs("data", exist_ok=True)

    filepath = os.path.join("data", uploaded_file.name)

    with open(filepath, "wb") as f:

        f.write(uploaded_file.read())

    full_text = extract_text_from_pdf(filepath)

    if not full_text.strip():

        st.warning("⚠ No readable text found in the PDF.")

        return

    if len(full_text) > 100_000:

        st.warning("⚠ PDF content too long. Using only first 100,000 characters.")

        full_text = full_text[:100_000]

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.markdown("### 📄 PDF Preview")

        display_pdf(filepath)

    with col2:

        st.markdown("### 🛠 Options")

        option = st.radio("Choose an option:", ["📄 Summarize", "❓ Q&A"])

        # Clear output and download options if user changes option

        if st.session_state.last_option != option:

            st.session_state.output = ""

            st.session_state.output_type = ""

            st.session_state.show_download_options = False

            st.session_state.selected_format = None

            st.session_state.last_option = option

        if option == "📄 Summarize":

            if st.button("Generate Summary"):

                with st.spinner("Generating summary..."):

                    summary = summarize_text(full_text)

                st.session_state.output = summary

                st.session_state.output_type = "summary"

                st.session_state.show_download_options = False

                st.session_state.selected_format = None

            if st.session_state.output_type == "summary" and st.session_state.output:

                st.subheader("📝 Summary")

                st.write(st.session_state.output)

                if not st.session_state.show_download_options:

                    if st.button("Download Summary"):

                        st.session_state.show_download_options = True

                if st.session_state.show_download_options:

                    selected = st.selectbox(

                        "Select download format",

                        ["txt", "doc", "csv"],

                        key="download_format_summary"

                    )

                    st.session_state.selected_format = selected

                    if st.session_state.selected_format:

                        href = get_download_link(

                            st.session_state.output,

                            f"summary.{st.session_state.selected_format}",

                            st.session_state.selected_format,

                        )

                        st.markdown(

                            f'<a href="{href}" download="summary.{st.session_state.selected_format}">'

                            f"Click here to download summary.{st.session_state.selected_format}</a>",

                            unsafe_allow_html=True,

                        )

        elif option == "❓ Q&A":

            qa_mode = st.radio("Choose Q&A Type:", ["🧠 Generate Questions", "🗨 Ask Your Question"])

            # Reset output if qa_mode changes

            if st.session_state.last_qa_mode != qa_mode:

                st.session_state.output = ""

                st.session_state.output_type = ""

                st.session_state.show_download_options = False

                st.session_state.selected_format = None

                st.session_state.last_qa_mode = qa_mode

            if qa_mode == "🧠 Generate Questions":

                num_qs = st.slider("Number of Questions", 1, 10, 3)

                if st.button("Generate Q&A"):

                    with st.spinner("Generating questions and answers..."):

                        result = generate_auto_qa(full_text, num_qs)

                    st.session_state.output = result

                    st.session_state.output_type = "auto_qa"

                    st.session_state.show_download_options = False

                    st.session_state.selected_format = None

                if st.session_state.output_type == "auto_qa" and st.session_state.output:

                    st.subheader("📚 Generated Q&A")

                    st.write(st.session_state.output)

                    if not st.session_state.show_download_options:

                        if st.button("Download Q&A"):

                            st.session_state.show_download_options = True

                    if st.session_state.show_download_options:

                        selected = st.selectbox(

                            "Select download format",

                            ["txt", "doc", "csv"],

                            key="download_format_auto_qa"

                        )

                        st.session_state.selected_format = selected

                        if st.session_state.selected_format:

                            href = get_download_link(

                                st.session_state.output,

                                f"auto_qa.{st.session_state.selected_format}",

                                st.session_state.selected_format,

                            )

                            st.markdown(

                                f'<a href="{href}" download="auto_qa.{st.session_state.selected_format}">'

                                f"Click here to download auto_qa.{st.session_state.selected_format}</a>",

                                unsafe_allow_html=True,

                            )

            elif qa_mode == "🗨 Ask Your Question":

                user_question = st.text_input("Enter your question:")

                if st.button("Get Answer") and user_question.strip() != "":

                    with st.spinner("Finding the answer..."):

                        result = generate_answer(full_text, user_question)

                    st.session_state.output = f"Q: {user_question}\nA: {result}"

                    st.session_state.output_type = "custom_qa"

                    st.session_state.show_download_options = False

                    st.session_state.selected_format = None

                if st.session_state.output_type == "custom_qa" and st.session_state.output:

                    st.subheader("💬 Answer")

                    q, a = st.session_state.output.split('\n', 1)

                    st.markdown(f"**{q}**")

                    st.markdown(a)

                    if not st.session_state.show_download_options:

                        if st.button("Download Answer"):

                            st.session_state.show_download_options = True

                    if st.session_state.show_download_options:

                        selected = st.selectbox(

                            "Select download format",

                            ["txt", "doc", "csv"],

                            key="download_format_custom_qa"

                        )

                        st.session_state.selected_format = selected

                        if st.session_state.selected_format:

                            href = get_download_link(

                                st.session_state.output,

                                f"custom_qa.{st.session_state.selected_format}",

                                st.session_state.selected_format,

                            )

                            st.markdown(

                                f'<a href="{href}" download="custom_qa.{st.session_state.selected_format}">'

                                f"Click here to download custom_qa.{st.session_state.selected_format}</a>",

                                unsafe_allow_html=True,

                            )

if __name__ == "__main__":

    main()
