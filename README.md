# GreenSightAI

<img width="598" alt="Logo" src="https://github.com/user-attachments/assets/f9099d98-55d3-4ae8-bfa0-4159c6d646a9" />


## 📄 **Project Description**

**GreenSightAI** is an intelligent ESG (Environmental, Social, and Governance) document assistant powered by OpenAI's GPT-4o-mini.  
It allows users to upload ESG PDF reports and ask questions in natural language. Answers are precise, contextual, and backed by content from the document.

This tool helps analysts, students, and sustainability professionals explore complex ESG disclosures efficiently. ⚡️📊

---

## 🚀 **Features**

- 📤 Upload ESG PDF reports  
- ❓ Ask natural language questions  
- 🔍 Get context-based AI answers  
- 📄 View pages used as sources  
- 🎧 Optional voice playback of answers  
- 🧪 Answer evaluation tab (demo)


## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/timmdkj/GreenSightAI.git
cd GreenSightAI

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# Install dependencies
pip install -r requirements-app.txt

# Run the Streamlit application
streamlit run app.py
```
## 🏗️ Project Architecture

- 📄 **PDF Processing**  
  Uploads and parses ESG PDF reports using `PyPDF2`. Text is split into overlapping chunks (512 tokens, 50 overlap).

- 🧠 **Embedding & Indexing**  
  Each chunk is embedded using `text-embedding-3-small` (OpenAI). A FAISS index is built locally and saved for fast similarity search.

- 🔎 **Retrieval & Reranking**  
  Top-k relevant chunks are retrieved using FAISS and re-ranked using GPT-3.5 prompts that score chunk relevance to the question (0–10).

- 🤖 **Answer Generation**  
  A final answer is generated using GPT-4o-mini based strictly on the most relevant chunks.

- 🎨 **User Interface**  
  Built with Streamlit, allowing users to upload PDFs, ask questions, view AI responses, explore source pages, and even listen to the answer via voice synthesis.

- 🧪 **Evaluation (demo)**  
  A simulated evaluation tab shows estimated reliability for the last answer (manually defined, pending full integration of automatic metrics).

---
