import os

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/PYTHON PROGRAMMING NOTES.pdf"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 10px;
        font-size: 18px;
    }

    .source-box {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📚 PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions from your Python Programming PDF</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK PDF
# ============================================================

if not os.path.exists(PDF_PATH):

    st.error(
        f"PDF not found: {PDF_PATH}"
    )

    st.stop()


# ============================================================
# LOAD PDF
# ============================================================

@st.cache_data
def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# CREATE CHUNKS
# ============================================================

@st.cache_data
def create_chunks(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    return model


# ============================================================
# CREATE FAISS INDEX
# ============================================================

@st.cache_resource
def create_faiss_index(chunks):

    embedding_model = load_embedding_model()

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# LOAD FLAN-T5
# ============================================================

@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL_NAME
    )

    return tokenizer, model


# ============================================================
# INITIALIZE RAG SYSTEM
# ============================================================

with st.spinner("Loading PDF..."):

    text = load_pdf(
        PDF_PATH
    )


if not text.strip():

    st.error(
        "No text could be extracted from the PDF."
    )

    st.stop()


with st.spinner("Creating document chunks..."):

    chunks = create_chunks(
        text
    )


with st.spinner("Loading embedding model..."):

    embedding_model = load_embedding_model()


with st.spinner("Creating FAISS index..."):

    index = create_faiss_index(
        chunks
    )


with st.spinner("Loading FLAN-T5 model..."):

    tokenizer, generator_model = load_llm()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document Information")

    st.write(
        "Python Programming Notes"
    )

    st.divider()

    st.write(
        f"📦 Total Chunks: **{len(chunks)}**"
    )

    st.write(
        f"🔢 Embedding Dimension: **{index.d}**"
    )

    st.write(
        f"🔎 Top K Results: **{TOP_K}**"
    )

    st.divider()

    st.header("⚙️ Models")

    st.write(
        f"Embedding: `{EMBEDDING_MODEL_NAME}`"
    )

    st.write(
        f"LLM: `{LLM_MODEL_NAME}`"
    )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask a question:",
    placeholder="Example: What are Python data types?"
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask Question",
    type="primary"
)


# ============================================================
# RAG PIPELINE
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # --------------------------------------------------------
    # QUESTION EMBEDDING
    # --------------------------------------------------------

    with st.spinner("Searching the document..."):

        question_embedding = embedding_model.encode(
            [question]
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype="float32"
        )


        # ----------------------------------------------------
        # FAISS SEARCH
        # ----------------------------------------------------

        distances, indices = index.search(
            question_embedding,
            TOP_K
        )


        # ----------------------------------------------------
        # CREATE CONTEXT
        # ----------------------------------------------------

        context_parts = []

        sources = []

        for rank, idx in enumerate(indices[0]):

            if idx < 0 or idx >= len(chunks):
                continue

            chunk = chunks[idx]

            distance = float(
                distances[0][rank]
            )

            context_parts.append(
                f"Context {rank + 1}:\n{chunk}"
            )

            sources.append(
                {
                    "rank": rank + 1,
                    "chunk": int(idx),
                    "distance": distance,
                    "text": chunk
                }
            )


        context = "\n\n".join(
            context_parts
        )


    # --------------------------------------------------------
    # RAG PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a document question answering system.

Use ONLY the information provided in the context.

Context:
{context}

Question:
{question}

Rules:
- Answer only using the context.
- Answer the question directly.
- Keep the answer short and clear.
- Do not use outside knowledge.
- Do not invent information.
- Do not copy unrelated sentences.
- If the answer is not available in the context, say:
"The answer is not available in the document."

Answer:
"""


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.spinner("Generating answer..."):

        outputs = generator_model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            do_sample=False
        )


        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()


    # ========================================================
    # DISPLAY FINAL ANSWER
    # ========================================================

    st.subheader("🤖 Answer")

    st.markdown(
        f"""
        <div class="answer-box">
        {answer}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # DISPLAY SOURCES
    # ========================================================

    st.subheader("📚 Retrieved Sources")

    for source in sources:

        with st.expander(
            f"Result {source['rank']} — "
            f"Chunk {source['chunk']} — "
            f"Distance {source['distance']:.4f}"
        ):

            st.write(
                source["text"]
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • FAISS • Sentence Transformers • FLAN-T5 • Streamlit"
)