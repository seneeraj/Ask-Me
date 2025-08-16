import streamlit as st
import fitz  # PyMuPDF
import easyocr
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Initialize models
reader = easyocr.Reader(['hi'])
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
qa_model = pipeline("text-generation", model="gpt2")  # Replace with Hindi-capable model if needed

# Helper: Extract text from PDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        page_text = page.get_text()
        if not page_text.strip():  # If empty, try OCR
            pix = page.get_pixmap()
            image_bytes = pix.tobytes("png")
            results = reader.readtext(image_bytes)
            page_text = " ".join([res[1] for res in results])
        text += page_text + "\n"
    return text

# Helper: Chunk and embed text
def chunk_and_embed(text, chunk_size=500):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return chunks, index

# Helper: Retrieve relevant chunks
def retrieve_chunks(query, chunks, index):
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, k=3)
    return [chunks[i] for i in I[0]]

# Helper: Generate answer
def generate_answer(query, context):
    prompt = f"प्रश्न: {query}\nउत्तर: {context}"
    response = qa_model(prompt, max_length=200, do_sample=True)[0]['generated_text']
    return response.split("उत्तर:")[-1].strip()

# Streamlit UI
st.set_page_config(page_title="Hindi Document Q&A", layout="wide")
st.title("📘 Hindi Document Q&A System")

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload First Hindi PDF", type=["pdf"])
with col2:
    file2 = st.file_uploader("Upload Second Hindi PDF (optional)", type=["pdf"])

query = st.text_input("🔍 Ask your question (in Hindi or English)")

if query and file1:
    with st.spinner("Processing documents..."):
        text1 = extract_text(file1)
        chunks1, index1 = chunk_and_embed(text1)

        if file2:
            text2 = extract_text(file2)
            chunks2, index2 = chunk_and_embed(text2)

    if "compare" in query.lower() or "difference" in query.lower() or "अंतर" in query:
        if file2:
            context1 = " ".join(retrieve_chunks(query, chunks1, index1))
            context2 = " ".join(retrieve_chunks(query, chunks2, index2))
            combined_context = f"दस्तावेज़ 1:\n{context1}\n\nदस्तावेज़ 2:\n{context2}\n\nइन दोनों के बीच अंतर बताइए।"
            answer = generate_answer(query, combined_context)
            st.subheader("📊 Comparison Answer")
            st.text_area("Answer", value=answer, height=300)
        else:
            st.warning("Please upload both documents to compare.")
    else:
        context = " ".join(retrieve_chunks(query, chunks1, index1))
        answer = generate_answer(query, context)
        st.subheader("📝 Answer from Document")
        st.text_area("Answer", value=answer, height=300)
