import os
import csv
import faiss
import torch
import numpy as np
import pandas as pd
import nltk
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from config_bert import OPENAI_API_KEY, COMPLETION_MODEL, BERT_MODEL_NAME, INDEX_DIR, CHUNK_SEPARATOR

# Ensure NLTK tokenizer is available
nltk.download('punkt')

# Init OpenAI client and BERT model
client = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
model = AutoModel.from_pretrained(BERT_MODEL_NAME)
model.eval()

# Load FAISS index and chunks
index = faiss.read_index(os.path.join(INDEX_DIR, "bert_optimized.index"))
with open(os.path.join(INDEX_DIR, "bert_optimized_chunks.txt"), "r", encoding="utf-8") as f:
    chunks = f.read().split(CHUNK_SEPARATOR)

# Load evaluation questions and references
raw_df = pd.read_csv("rag_evaluation_dataset.csv")

# Compute BERT embedding
def get_query_embedding(query):
    with torch.no_grad():
        encoded = tokenizer([query], return_tensors="pt", truncation=True, padding=True)
        output = model(**encoded)
        return output.last_hidden_state.mean(dim=1).squeeze().numpy().astype("float32")

# Search in FAISS
def search_top_k(query_embedding, k=10):
    _, indices = index.search(query_embedding.reshape(1, -1), k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# Re-rank with GPT
def rerank_and_select(query, retrieved_chunks, top_n=3):
    prompt = f"""You are given a question and a set of chunks. Rank the chunks from most to least relevant to answer the question.

Question: {query}

Chunks:"""
    for i, chunk in enumerate(retrieved_chunks):
        prompt += f"\n[{i}] {chunk}"
    prompt += f"\n\nReturn only the top {top_n} indices, comma-separated. Example: 2, 1, 0"

    res = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    ranked = res.choices[0].message.content
    selected_indices = [int(i.strip()) for i in ranked.split(",") if i.strip().isdigit() and int(i.strip()) < len(retrieved_chunks)]
    return [retrieved_chunks[i] for i in selected_indices[:top_n]]

# Generate GPT response
def generate_response(query, top_chunks):
    prompt = f"Answer the following question using the context below:\n\n" + "\n---\n".join(top_chunks) + f"\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Loop over dataset
results = []
for idx, row in raw_df.iterrows():
    question = row["question"]
    reference = row["reference"]

    query_emb = get_query_embedding(question)
    top_chunks = search_top_k(query_emb, k=10)
    reranked_chunks = rerank_and_select(question, top_chunks, top_n=3)
    gpt_answer = generate_response(question, reranked_chunks)

    results.append({
        "user_input": question,
        "retrieved_contexts": str(reranked_chunks),
        "response": gpt_answer,
        "reference": reference
    })
    print(f"✅ Processed question {idx+1}/{len(raw_df)}")

# Save the optimized dataset
output_file = "rag_evaluation_dataset_optimized.csv"
with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["user_input", "retrieved_contexts", "response", "reference"], delimiter=";")
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Optimized dataset saved to {output_file}")
