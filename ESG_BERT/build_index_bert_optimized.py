# build_index_bert_optimized.py

import os
import faiss
import numpy as np
import PyPDF2
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from config_bert import PDF_PATH, INDEX_DIR, CHUNK_SEPARATOR, BERT_MODEL_NAME

# Download NLTK punkt tokenizer
nltk.download("punkt")

# Ensure FAISS directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
model = AutoModel.from_pretrained(BERT_MODEL_NAME)
model.eval()

# Extract text from PDF
def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    for page in reader.pages:
        yield page.extract_text() or ""

# Chunk text by sentences (grouped under token limit)
def chunk_by_sentences(text, max_tokens=150):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if current_len + len(tokens) > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = len(tokens)
        else:
            current_chunk.append(sentence)
            current_len += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Generate BERT embeddings
def get_bert_embedding(texts):
    with torch.no_grad():
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**encoded)
        return outputs.last_hidden_state.mean(dim=1).numpy()

# Process the PDF and build index
chunks = []
embeddings = []

for page_text in extract_text_from_pdf(PDF_PATH):
    for chunk in chunk_by_sentences(page_text):
        chunks.append(chunk)
        emb = get_bert_embedding([chunk])
        embeddings.append(emb[0])

# Build and save FAISS index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, os.path.join(INDEX_DIR, "bert_optimized.index"))

# Save chunks
with open(os.path.join(INDEX_DIR, "bert_optimized_chunks.txt"), "w", encoding="utf-8") as f:
    f.write(CHUNK_SEPARATOR.join(chunks))

print("✅ Optimized BERT FAISS index built successfully.")
