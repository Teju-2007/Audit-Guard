# 🛡️ Audit-Guard: AI-Powered Whitepaper Auditor

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://audit-guard-hyzugm25zfjq4bkgzkhmhw.streamlit.app/)
[![LLM](https://img.shields.io/badge/Groq-GPT--OSS--120B-f05123?style=for-the-badge)](https://groq.com)
[![Vector DB](https://img.shields.io/badge/Pinecone-Hybrid%20RAG-000000?style=for-the-badge&logo=pinecone)](https://pinecone.io)

**Audit-Guard** is a high-performance RAG (Retrieval-Augmented Generation) intelligence agent tailored for Venture Capitalists, Web3 Analysts, and Regulatory Compliance Officers. It analyzes technical crypto whitepapers (including MiCA Title II compliance) to flag technical gaps, regulatory moats, architecture flaws, and decentralization risks.

---

## 🚀 Key Features

* **Executive Scoring Dashboard:** Generates instant 0–100 metrics for **Technical Maturity**, **Regulatory Moat**, and **Decentralization**, accompanied by top critical red flags.
* **Multi-Stage RAG Pipeline:** Combines dense semantic search via `all-MiniLM-L6-v2` with Pinecone vector storage and re-ranks top candidates using Pinecone's `bge-reranker-v2-m3` model for high-precision context retrieval.
* **Deep Reasoning Engine:** Powered by Groq's high-speed LPU inference running `openai/gpt-oss-120b` for technical paper analysis.
* **Auditable Citations:** Every chat response provides page-level source references and expandable text snippets for factual verification.
* **Deal Memo Export:** One-click download of structured executive deal memos (`.txt`) for investment committee distribution.

---

## 🛠️ Tech Stack

* **LLM Engine:** Groq API (`openai/gpt-oss-120b`)
* **Vector Database:** Pinecone
* **Re-ranker:** Pinecone Inference API (`bge-reranker-v2-m3`)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Frontend:** Streamlit
* **PDF Parser:** PyPDF
* **Environment & Config:** `python-dotenv`

---

## 🏗️ Architecture & Workflow

```
[ PDF Whitepaper ] 
        │
        ├──> Text Extraction (PyPDF)
        ├──> Smart Sampling ──> Scorecard Generation (Groq GPT-OSS 120B)
        │
        └──> Chunking (500 chars) ──> Batch Encoding (all-MiniLM-L6-v2) ──> Upsert to Pinecone Index
                                                                                     │
[ User Query ] ──> Vector Search (Top-20) ──> Rerank (bge-reranker-v2-m3 Top-5) ───┴──> Groq 120B RAG Audit Output
```

---

## 💻 Local Setup & Installation

### 1. Prerequisites
* Python 3.10 or higher
* Groq API Key
* Pinecone API Key (with an index named `audit-guard-index` configured for **384 dimensions** and **cosine similarity**)

### 2. Clone the Repository
```bash
git clone [https://github.com/Teju-2007/Audit-Guard.git](https://github.com/Teju-2007/Audit-Guard.git)
cd Audit-Guard
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## 📦 Requirements (`requirements.txt`)

```text
streamlit
groq
pinecone-client
pypdf
sentence-transformers
python-dotenv
```
