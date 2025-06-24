import streamlit as st
import os
import fitz  # PyMuPDF
from docx import Document
import joblib
import tempfile
import pandas as pd
import requests

# === Load trained model ===
model = joblib.load("customs_doc_classifier.pkl")

# === LibreTranslate API
def libre_translate(text, target_lang):
    url = "https://libretranslate.de/translate"
    payload = {"q": text, "source": "auto", "target": target_lang, "format": "text"}
    try:
        response = requests.post(url, data=payload)
        return response.json()["translatedText"] if response.status_code == 200 else "⚠️ Translation failed."
    except Exception as e:
        return f"⚠️ API error: {e}"

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

# === Page Config ===
st.set_page_config(page_title="Smart Document Classifier", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #0073e6;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stSuccess {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# === App Title ===
st.title("📄 Smart Customs Document Classifier")

# === Tabs ===
tab1, tab2 = st.tabs(["🧠 Classify Document", "🌍 Translate Text"])

with tab1:
    st.subheader("📤 Upload a document")
    uploaded_file = st.file_uploader("Upload a document (PDF or DOCX)", type=["pdf", "docx"])

    if uploaded_file:
        st.info(f"**Filename:** `{uploaded_file.name}`")
        extracted_text = extract_text(uploaded_file)

        if extracted_text.strip():
            st.markdown("### 📝 Extracted Text Preview")
            st.text_area("Text Preview", extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text, height=250)

            if st.button("🔍 Classify Document"):
                label = model.predict([extracted_text])[0]
                st.success(f"✅ Predicted Document Type: **{label.upper()}**")
        else:
            st.error("❌ No readable text found in the document.")

with tab2:
    st.subheader("🌐 Translate Any Text")
    text_to_translate = st.text_area("Enter text to translate:", height=200)
    target_lang = st.selectbox("🌍 Select Target Language", ["en", "es", "fr", "de", "hi", "ja", "zh", "ar"])

    if st.button("🌍 Translate & Download Report"):
        if not text_to_translate.strip():
            st.warning("Please enter some text.")
        else:
            translated_text = libre_translate(text_to_translate, target_lang)
            st.text_area("Translated Text:", translated_text, height=200)

            # Download CSV
            report_df = pd.DataFrame({
                "Original Text": [text_to_translate],
                "Translated Text": [translated_text],
                "Target Language": [target_lang]
            })
            report_csv = report_df.to_csv(index=False)
            st.download_button(
                "📥 Download Translation Report",
                data=report_csv,
                file_name="translation_report.csv",
                mime="text/csv"
            )

# === Footer ===
st.markdown("---")
st.caption("Made with ❤️ using Streamlit, scikit-learn and LibreTranslate")
