import streamlit as st
import os
import fitz  # PyMuPDF
from docx import Document
import joblib
import tempfile

# === Load trained model ===
model = joblib.load("customs_doc_classifier.pkl")

# === Text extraction functions ===
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return " ".join(page.get_text() for page in doc)

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    if ext == ".pdf":
        return extract_text_from_pdf(tmp_file_path)
    elif ext == ".docx":
        return extract_text_from_docx(tmp_file_path)
    else:
        st.warning(f"⚠️ Unsupported file type: {ext}")
        return ""

# === Streamlit UI ===
st.set_page_config(page_title="Smart Document Classifier", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        padding: 0.5em 2em;
        border-radius: 10px;
    }
    .stSuccess {
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 Smart Document Classifier")
st.markdown("#### Upload a document and let AI identify its type!")

st.markdown("---")

uploaded_file = st.file_uploader(
    "📤 Upload a document (PDF or Word)", 
    type=["pdf", "docx"]
)

if uploaded_file:
    st.markdown("### 📁 File Details")
    st.info(f"**Filename:** `{uploaded_file.name}`")

    extracted_text = extract_text(uploaded_file)

    if extracted_text.strip():
        st.markdown("---")
        st.markdown("### 📝 Extracted Text Preview")
        st.text_area("Text Preview", extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text, height=300)

        # Predict label
        label = model.predict([extracted_text])[0]

        st.markdown("---")
        st.markdown("### 🔍 Classification Result")
        st.success(f"✅ **Predicted Document Type:** `{label.upper()}`")
    else:
        st.error("❌ No readable text found in the document.")
