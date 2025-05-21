# GreenSight AI - ESG Assistant

import os
import openai
import streamlit as st
import yaml
from pathlib import Path
import base64
import faiss
import numpy as np
import re
from PyPDF2 import PdfReader
import streamlit.components.v1 as components

# === CONFIGURATION ===
def load_config():
    config_path = os.path.abspath("config/openai_config_template.yaml")
    if not os.path.exists(config_path):
        st.error(f"❌ Configuration file not found at: {config_path}")
        st.stop()
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
openai.api_key = config['openai_api_key']
client = openai.OpenAI(api_key=openai.api_key)

EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4o-mini"
CHUNK_SEPARATOR = "\n---\n"

# === UTILS ===
def clean_filename(name):
    return Path(name).stem.replace(" ", "_").replace(".", "_").lower()

def get_embedding(text):
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32)

def parse_score(score_str):
    match = re.search(r"(\d+)(?:\s*/\s*10)?", score_str)
    if match:
        return float(match.group(1))
    return 0.0

def split_text_into_chunks(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# === INDEXING PDF ===
def build_index_from_pdf(file_path):
    filename = clean_filename(Path(file_path).name)
    folder = Path("data") / filename
    folder.mkdir(parents=True, exist_ok=True)

    chunks_file = folder / "chunks.txt"
    index_file = folder / "index.faiss"

    if chunks_file.exists() and index_file.exists():
        index = faiss.read_index(str(index_file))
        with open(chunks_file, "r", encoding="utf-8") as f:
            raw = f.read().split(CHUNK_SEPARATOR)
        chunks = [x.split("|page:")[0] for x in raw]
        pages = [int(x.split("|page:")[-1]) for x in raw]
        return chunks, pages, index

    reader = PdfReader(file_path)
    chunks, pages = [], []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for chunk in split_text_into_chunks(text):
            chunks.append(chunk)
            pages.append(i + 1)

    embeddings = np.array([get_embedding(chunk) for chunk in chunks])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write(CHUNK_SEPARATOR.join([f"{chunk}|page:{page}" for chunk, page in zip(chunks, pages)]))

    faiss.write_index(index, str(index_file))

    return chunks, pages, index

# === RERANK + ANSWER ===
def rerank_contexts(question, chunks, pages, index, top_k=5, faiss_k=20):
    query_emb = get_embedding(question)
    _, I = index.search(query_emb.reshape(1, -1), faiss_k)
    candidates = [(chunks[i], pages[i]) for i in I[0] if i < len(chunks)]

    scored = []
    for chunk, page in candidates:
        prompt = f"Rate how relevant this chunk is to the question (0–10):\n\nQuestion: {question}\nChunk: {chunk[:1000]}\n\nScore:"
        try:
            score_resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score_str = score_resp.choices[0].message.content.strip()
            score = parse_score(score_str)
            scored.append((chunk, page, score))
        except Exception as e:
            scored.append((chunk, page, 0.0))

    reranked = sorted(scored, key=lambda x: -x[2])[:top_k]
    top_chunks = [r[0] for r in reranked]
    top_pages = [r[1] for r in reranked]
    return top_chunks, top_pages

def generate_answer(contexts, question):
    prompt = (
        "You are a sustainability analyst helping a user explore an ESG report. "
        "Base your answer strictly on the provided context. "
        "If the context clearly includes the answer, provide it. "
        "Context:\n"
        + "\n---\n".join(contexts)
        + f"\n\nQuestion: {question}\nAnswer:"
    )
    response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# === STYLING ===
st.set_page_config(page_title="GreenSight AI", layout="centered")
st.markdown("""
    <style>
        body, .stApp {
            background-color: white !important;
            color: black;
            font-family: 'Arial', sans-serif;
        }
        .block-container {
            padding-top: 3rem;
        }
        h1, h2, h3, h4 {
            color: #1a1a1a;
        }
        .btn-custom {
            font-size: 15px;
            padding: 10px 18px;
            border-radius: 8px;
            background-color: #f2f2f2;
            border: 1px solid #ccc;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# === LOGO ===
def display_logo():
    logo_path = "assets/Logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 40px; margin-bottom: 0px;">
                <img src="data:image/png;base64,{encoded}" width="220">
                <p style="font-size:16px; color: #4d4d4d; margin-top: 8px;">
                    <em>Empowering ESG decisions through AI</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# === NAVIGATION ===
if "home_passed" not in st.session_state:
    st.session_state.home_passed = False

if not st.session_state.home_passed:
    st.markdown("<br><br>", unsafe_allow_html=True)
    display_logo()
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <h3 style='text-align: center; font-weight: 400;'>Welcome to GreenSight AI, your ESG copilot</h3>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Enter GreenSight AI", use_container_width=True):
        st.session_state.home_passed = True
    st.stop()

st.sidebar.title("GreenSight AI")
page = st.sidebar.radio("Go to", ["Assistant", "Evaluation", "About"])

# === ASSISTANT ===
if page == "Assistant":
    display_logo()
    st.header("Ask your ESG PDF")
    st.markdown("""
        <p style='color: grey; font-size: 15px;'>
        Upload your ESG report and ask any question about its content.<br>
        The assistant will only answer based on the content of the document.
        </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your ESG PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing and indexing your document..."):
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            chunks, pages, index = build_index_from_pdf(file_path)
            st.session_state.chunks = chunks
            st.session_state.pages = pages
            st.session_state.index = index
            st.session_state.filename = uploaded_file.name
            st.success("Document ready!")

    if "index" in st.session_state and "chunks" in st.session_state:
        example_question = "What are the company’s emissions targets for 2030?"
        st.text_input(
            "Ask a question about your document:",
            value=st.session_state.get("current_question", ""),
            placeholder=example_question,
            key="question_input",
            on_change=lambda: st.session_state.update({"current_question": st.session_state.question_input})
        )

        if st.button("Ask", use_container_width=True):
            question = st.session_state.current_question
            with st.spinner("Thinking..."):
                top_chunks, top_pages = rerank_contexts(question, st.session_state.chunks, st.session_state.pages, st.session_state.index)
                answer = generate_answer(top_chunks, question)
                st.session_state.last_question = question
                st.session_state.last_response = answer
                st.session_state.last_contexts = top_chunks
                st.session_state.last_pages = top_pages

        if "last_response" in st.session_state:
            st.markdown("---")
            st.subheader("Assistant Answer")
            st.markdown(f"""
            <div style="background-color: #f9f9f9; padding: 15px 20px; border-radius: 10px; border: 1px solid #ddd; font-size: 16px;">
                {st.session_state.last_response}
            </div>
            """, unsafe_allow_html=True)

            response_text = st.session_state.last_response.strip().lower()
            keywords = ["not provide", "not mentioned", "no information", "unable to find", "does not", "does not include"]
            if not any(keyword in response_text for keyword in keywords):
                pages = st.session_state.get("last_pages", [])
                if pages:
                    unique_pages = sorted(set(pages))
                    st.caption(f"📄 Context retrieved from pages: {', '.join(map(str, unique_pages))}")

            components.html(
                f"""
                <script>
                    var synth = window.speechSynthesis;
                    var utterance;
                    function toggleSpeech(text) {{
                        if (synth.speaking) {{
                            synth.cancel();
                        }} else {{
                            utterance = new SpeechSynthesisUtterance(text);
                            utterance.lang = 'en-US';
                            utterance.rate = 0.70;
                            synth.speak(utterance);
                        }}
                    }}
                </script>
                <div style="margin-top:10px;">
                    <button onclick="toggleSpeech(`{st.session_state.last_response}`)" class="btn-custom">
                        🎧 Listen
                    </button>
                </div>
                """,
                height=80,
            )

# === EVALUATION ===
# This evaluation section shows a static reliability score used for demonstration purposes.
# The value (e.g., 91%) is not dynamically computed by the app.
# It was manually defined as an example based on internal testing,
# averaging the "Faithfulness" and "Answer Relevancy" scores obtained for a sample ESG question.

elif page == "Evaluation":
    display_logo()
    st.header("ESG Assistant Evaluation")
    question = st.session_state.get("last_question", "")
    answer = st.session_state.get("last_response", "")

    if not question or not answer:
        st.info("Please ask a question in the Assistant tab first")
    else:
        st.markdown(f"**Question:** {question}")
        st.markdown(f"**Answer:** {answer}")
        if st.button("🔍 Evaluate this answer"):
            st.success("The answer is considered reliable.")
            st.metric(label="Estimated Reliability", value="91%")
            st.progress(0.91)
            st.caption("This is a simulated evaluation for demonstration purposes.")

# === PAGE 3: ABOUT ===
elif page == "About":
    display_logo()
    st.header("About this Project")
    st.markdown("""
    **GreenSight AI** is an intelligent assistant designed to support ESG (Environmental, Social, and Governance) analysis through AI-driven document understanding.

    It addresses the challenge of navigating long, complex ESG reports by making their content directly accessible through natural language questions.

    It allows users to explore sustainability reports and extract reliable, contextual answers to ESG-related questions.

    Key features include:
    - Fast question-answering from corporate ESG documents  
    - Highlighting the relevant sections used as sources  
    - A user-friendly interface for interactive exploration  
    - (Coming soon) Evaluation features to assess the quality and reliability of answers

    The goal is to empower analysts, students, and sustainability professionals to interact with ESG information more easily and efficiently.

    This application was developed as part of the **VO2 x FTD – AI for ESG** program.
    """)
