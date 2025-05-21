# GreenSightAI

<img width="598" alt="Logo" src="https://github.com/user-attachments/assets/f9099d98-55d3-4ae8-bfa0-4159c6d646a9" />


## 📄 **Project Description**

**GreenSightAI** is an intelligent ESG (Environmental, Social, and Governance) document assistant built using **Retrieval-Augmented Generation (RAG)** technology.  
It enables users to upload ESG reports in PDF format and ask natural language questions, delivering **precise and contextually relevant answers** supported by excerpts from the documents.

Powered by **OpenAI's GPT-4o-mini** and **LlamaIndex**, GreenSightAI helps investors, analysts, and sustainability professionals analyze complex ESG reports **faster and more accurately**. ⚡️📊

---

## 🚀 **Features**

- 📂 Upload and index ESG PDF reports  
- ❓ Ask detailed, context-aware questions about the document content  
- 🔍📄 View the source passages and page numbers that support each answer  
- 💡 Explanation of AI-generated answers with "Why this answer?" feature  
- 🧪 Planned features: automated evaluation with RAGAS, multi-document support, dashboard interface  

---

## 🛠️ **Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/GreenSightAI.git
cd GreenSightAI

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
