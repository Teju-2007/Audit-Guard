import streamlit as st
import os
import uuid
import json
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# 1. INITIAL SETUP
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Use wide layout to prevent text squeezing
st.set_page_config(page_title="Audit-Guard", page_icon="🛡️", layout="wide")

# --- UI NORMALIZATION (CSS) ---
st.markdown("""
    <style>
    /* Force normal font size for all paragraph text */
    html, body, [class*="st-"], .stMarkdown p {
        font-size: 15px !important;
        line-height: 1.6 !important;
        color: #333;
    }
    /* Shrink the headers to professional levels */
    h1 { font-size: 26px !important; color: #1E3A8A; margin-bottom: 20px !important; }
    h2 { font-size: 20px !important; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-top: 25px !important; }
    h3 { font-size: 17px !important; font-weight: bold !important; color: #1E3A8A; }
    
    /* Normalize Metric Dashboard fonts */
    [data-testid="stMetricValue"] { font-size: 24px !important; color: #1E3A8A; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; font-weight: 600 !important; text-transform: uppercase; }

    /* Compact Chat Messages */
    .stChatMessage { padding: 1rem !important; margin-bottom: 0.5rem !important; border-radius: 10px; border: 1px solid #f0f2f6; }
    
    /* Info box styling */
    .stInfo { font-size: 14px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DASHBOARD SCORING FUNCTION ---
def get_audit_scorecard(context_summary, groq_client):
    scoring_system = f"""
    Analyze the following technical context from a crypto whitepaper. 
    Provide 3 scores (0-100) and identify the top 3 'Red Flags'.
    0 = Critical Risk/Copied Tech, 100 = Industry Leading/Fully Compliant.

    Return ONLY a JSON object:
    {{
        "tech": int,
        "reg": int,
        "dec": int,
        "flags": ["list of strings"]
    }}
    
    Context: {context_summary}
    """
    try:
        resp = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": scoring_system}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"tech": 0, "reg": 0, "dec": 0, "flags": [f"Scorecard Error: {str(e)}"]}

# 2. INITIALIZE MODELS
@st.cache_resource
def init_models():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("audit-guard-index")
        groq_client = Groq(api_key=GROQ_API_KEY)
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        return pc, index, groq_client, embed_model
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

pc, index, groq_client, embed_model = init_models()

# 3. SIDEBAR & INGESTION
st.sidebar.header("📁 Project Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Whitepaper", type="pdf")

if uploaded_file:
    if st.sidebar.button("🚀 Run Full Audit"):
        with st.spinner("Indexing & Scoring..."):
            # Clean Slate
            index.delete(delete_all=True)
            reader = PdfReader(uploaded_file)
            vectors_to_upsert = []
            all_text = ""

            # Single-pass indexing
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                all_text += page_text + " "
                
                # Chunking (500 chars with 100 char overlap)
                chunks = [page_text[i:i+500] for i in range(0, len(page_text), 400)]
                for chunk in chunks:
                    vectors_to_upsert.append({
                        "id": str(uuid.uuid4()),
                        "values": embed_model.encode(chunk).tolist(),
                        "metadata": {"text": chunk, "page": page_num + 1}
                    })

            # Batch Upsert
            for i in range(0, len(vectors_to_upsert), 100):
                index.upsert(vectors=vectors_to_upsert[i : i + 100])
            
            # --- SMART SAMPLING FOR SCORECARD ---
            first_pages = all_text[:2000]
            middle_pages = all_text[len(all_text)//2 : len(all_text)//2 + 2000]
            end_pages = all_text[-2000:]
            smart_summary_context = f"{first_pages}\n{middle_pages}\n{end_pages}"
            
            # Generate Scorecard
            st.session_state.scorecard = get_audit_scorecard(smart_summary_context, groq_client)
            st.sidebar.success(f"Audit Ready! ({len(vectors_to_upsert)} nodes)")

# 4. MAIN UI DISPLAY
st.title("🛡️ Audit-Guard: Executive Dashboard")

if "scorecard" in st.session_state:
    s = st.session_state.scorecard
    
    # Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Tech Maturity", f"{s.get('tech', 0)}/100")
    col2.metric("Regulatory Moat", f"{s.get('reg', 0)}/100")
    col3.metric("Decentralization", f"{s.get('dec', 0)}/100")
    
    # Red Flags Expander
    with st.expander("🚩 Critical Technical Red Flags", expanded=True):
        flags = s.get('flags', [])
        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.success("No immediate red flags detected.")
            
    # FIXED: Safe Report Generation for Download
    report_content = f"""
AUDIT REPORT: {uploaded_file.name}
----------------------------------
TECH SCORE: {s.get('tech', 0)}/100
REGULATORY SCORE: {s.get('reg', 0)}/100
DECENTRALIZATION: {s.get('dec', 0)}/100

CRITICAL RED FLAGS:
{chr(10).join(['- ' + str(f) for f in s.get('flags', [])])}
    """
    st.sidebar.download_button(
        label="📥 Download Deal Memo", 
        data=report_content, 
        file_name=f"audit_{uuid.uuid4().hex[:5]}.txt",
        mime="text/plain"
    )

st.markdown("---")

# 5. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. RAG CHAT LOGIC
if prompt := st.chat_input("Deep dive into specific technical risks..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing document..."):
        # A. Semantic Search
        query_vec = embed_model.encode(prompt).tolist()
        results = index.query(vector=query_vec, top_k=20, include_metadata=True)
        
        # B. Rerank Phase
        docs_to_rerank = [
            {"id": r['id'], "text": r['metadata']['text'], "metadata": r['metadata']} 
            for r in results.get('matches', [])
        ]
        
        if docs_to_rerank:
            rerank_res = pc.inference.rerank(
                model="bge-reranker-v2-m3", 
                query=prompt, 
                documents=docs_to_rerank, 
                top_n=5, 
                return_documents=True
            )
            
            # C. Build Context & Citations
            context_chunks = []
            citations = []
            for hit in rerank_res.data:
                txt = hit.document['text'] if isinstance(hit.document, dict) else hit.document.text
                meta = hit.document['metadata'] if isinstance(hit.document, dict) else hit.document.metadata
                context_chunks.append(txt)
                citations.append(f"Page {int(meta.get('page', 0))} (Match: {hit.score:.2f})")
            
            full_context = "\n\n---\n\n".join(context_chunks)
            
            # D. Expert Analysis Prompt (Compact Output)
            system_prompt = """
            You are a Senior VC Auditor. Keep your output professional, grounded, and COMPACT.
            DO NOT USE # OR ## HEADERS. 
            Instead, use **BOLD UPPERCASE** for section headers.
            
            Structure:
            **DIRECT ANSWER**
            (Text here)
            
            **TECHNICAL EVIDENCE**
            (Bullet points with citations)
            
            **RISKS & GAPS**
            (Bullet points or 'NOT DISCLOSED')
            """
            
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CONTEXT:\n{full_context}\n\nQUESTION: {prompt}"}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )
            
            response = chat_completion.choices[0].message.content
            
            # E. Assistant Display
            with st.chat_message("assistant"):
                st.markdown(response)
                with st.expander("📚 View Technical Sources"):
                    for i, chunk in enumerate(context_chunks):
                        st.caption(f"**Source {i+1}: {citations[i]}**")
                        st.info(chunk)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("No relevant information found. Please ensure the document is indexed.")
