# GreenSightAI

[![Watch the demo](https://img.youtube.com/vi/b0mpqq0c4K4/maxresdefault.jpg)](https://www.youtube.com/watch?v=b0mpqq0c4K4)

Click the thumbnail above to watch the demo of GreenSight AI on YouTube.

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

## 📁 Project Structure
```text
FTD_ESG_ASSISTANT/
├── ESG_BERT/                             # Scripts or experiments using BERT for ESG embeddings
├── asset/                                # Static assets like logo
│   └── Logo.png
├── data/                                 # Contains indexed ESG documents and evaluation databases
│   ├── totalenergies/                    # Directory for TotalEnergies report
│   │   ├── totalenergies.pdf             # Original uploaded ESG report
│   │   ├── chunks.txt                    # Text chunks with page info (used in FAISS indexing)
│   │   └── index.faiss                   # FAISS index built from the chunks
│   └── Reports_Evaluation_Databases_Drive/  # Annotated evaluation databases for RAGAS and QA tests
├── app.py                                # Main Streamlit app (GreenSight AI)
├── RAG_Model_Evaluation_Notebook.ipynb   # Notebook to test RAGAS evaluation
├── requirements-app.txt                  # Minimal list of required packages
├── README.md                             # Main documentation file
```
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

---

## 📊 RAGAS Evaluation

We evaluated the quality of answers using both **automated metrics** and **manual (human) evaluation**.

✅ Specifically, we focused on two key RAGAS metrics:  
- **Faithfulness**: Is the answer grounded in the retrieved context?  
- **Answer Relevancy**: Does the answer directly address the user's question?

👥 Human evaluation was also conducted to validate and complement the automated scores.

The best-performing evaluation configuration is currently integrated into the app (demo only).

📓 For details on all tested methods and full evaluation results, see the notebook:  
`RAG_Model_Evaluation_Notebook.ipynb`

## 🔮 Future Work

- **Automated Evaluation Integration**  
  Fully integrate RAGAS-based answer evaluation into the app (currently a static demo), allowing real-time scoring of faithfulness and relevancy for each answer.

- **Table Extraction Improvements**  
  Improve extraction and processing of tables from ESG reports to better incorporate tabular data into the question-answering workflow.

