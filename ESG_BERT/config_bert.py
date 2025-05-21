# config_bert.py

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COMPLETION_MODEL = "gpt-3.5-turbo"  # pour les réponses dans RAGAS
BERT_MODEL_NAME = "nbroad/ESG-BERT"

CHUNK_SIZE = 512
CHUNK_SEPARATOR = "\n===CHUNK_SEPARATOR===\n"
INDEX_DIR = "faiss_indices"
PDF_PATH = "C:/Users/moham/OneDrive/Bureau/ESG_AI_AssISTANT/totalenergies_report.pdf"
