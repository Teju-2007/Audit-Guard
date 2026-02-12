# 🛡️ Audit-Guard: AI-Powered Whitepaper Auditor

Audit-Guard Pro is a high-performance **RAG (Retrieval-Augmented Generation)** agent designed for Venture Capitalists and Regulatory Compliance officers. It analyzes crypto whitepapers (MiCA Title II) to identify technical gaps, regulatory moats, and decentralization risks.

### 🚀 Key Features
* **Smart Scoring**: Generates 0-100 scores for Tech, Reg, and Decentralization.
* **Ruthless Auditor Persona**: Specialized LLM prompting for risk detection.
* **Multi-Stage Retrieval**: Uses Pinecone Vector Search + BGE-Reranker for high-precision evidence.
* **Executive Dashboard**: Built with Streamlit for a sleek, compact UI.

### 🛠️ Tech Stack
* **LLM**: Groq (Llama 3.3 70B)
* **Vector DB**: Pinecone
* **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
* **Orchestration**: Streamlit & PyPDF