from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os


# ============================================================
# PDF RAG QUESTION ANSWERING SYSTEM
# ============================================================

PDF_PATH = "data/PYTHON PROGRAMMING NOTES.pdf"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3


# ============================================================
# 1. LOAD PDF
# ============================================================

print("\n" + "=" * 60)
print("PDF RAG QUESTION ANSWERING SYSTEM")
print("=" * 60)

print("\n[1/6] Loading PDF...")

if not os.path.exists(PDF_PATH):
    print(f"ERROR: PDF not found at: {PDF_PATH}")
    exit()

reader = PdfReader(PDF_PATH)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

if not text.strip():
    print("ERROR: No text could be extracted from PDF.")
    exit()

print("PDF Text Extracted Successfully!")


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

print("\n[2/6] Creating chunks...")

chunks = []

start = 0

while start < len(text):

    end = start + CHUNK_SIZE

    chunk = text[start:end]

    if chunk.strip():
        chunks.append(chunk.strip())

    start = end - CHUNK_OVERLAP

print("Total Chunks:", len(chunks))


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("\n[3/6] Loading Embedding Model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding Model Loaded!")


# ============================================================
# 4. CREATE EMBEDDINGS + FAISS
# ============================================================

print("\n[4/6] Creating Embeddings...")

embeddings = embedding_model.encode(
    chunks,
    show_progress_bar=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print("Embeddings Created!")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS Index Created!")

print("Embedding Dimension:", dimension)
print("FAISS Vectors:", index.ntotal)


# ============================================================
# 5. LOAD LOCAL LLM
# ============================================================

print("\n[5/6] Loading FLAN-T5 Model...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

generator_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

print("FLAN-T5 Model Loaded!")


# ============================================================
# 6. QUESTION ANSWERING LOOP
# ============================================================

print("\n[6/6] RAG System Ready!")

print("\n" + "=" * 60)
print("Ask questions about your PDF.")
print("Type 'exit' to stop.")
print("=" * 60)


while True:

    # --------------------------------------------------------
    # Get question
    # --------------------------------------------------------

    question = input("\nAsk a question: ").strip()

    if question.lower() == "exit":
        print("\nRAG System stopped.")
        break

    if not question:
        print("Please enter a question.")
        continue


    # --------------------------------------------------------
    # Question Embedding
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )


    # --------------------------------------------------------
    # FAISS Search
    # --------------------------------------------------------

    distances, indices = index.search(
        question_embedding,
        TOP_K
    )


    # --------------------------------------------------------
    # Build Context
    # --------------------------------------------------------

    context_parts = []
    sources = []

    print("\n" + "-" * 60)
    print("RETRIEVED CHUNKS")
    print("-" * 60)

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
                "distance": distance
            }
        )

        print(
            f"\n--- Result {rank + 1} ---"
        )

        print(
            f"Chunk: {idx}"
        )

        print(
            f"Distance: {distance:.4f}"
        )

        print(chunk)


    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # RAG Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document question answering system.

Answer the question using ONLY the information provided
in the context.

Context:
{context}

Question:
{question}

Rules:
- Use only information from the context.
- Answer the question directly.
- Keep the answer short and clear.
- Do not use outside knowledge.
- Do not copy unrelated sentences.
- Do not invent information.
- If the answer is not present in the context, answer:
"The answer is not available in the document."

Answer:
"""


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    # --------------------------------------------------------
    # Generate Answer
    # --------------------------------------------------------

    outputs = generator_model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        do_sample=False
    )


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


    # --------------------------------------------------------
    # Display Answer
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)


    # --------------------------------------------------------
    # Display Sources
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("SOURCES")
    print("-" * 60)

    for source in sources:

        print(
            f"Result {source['rank']} | "
            f"Chunk {source['chunk']} | "
            f"Distance {source['distance']:.4f}"
        )

    print("=" * 60)