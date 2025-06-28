import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
from keybert import KeyBERT

# ------------------------
# Load Models Once
# ------------------------
@st.cache_resource
def load_models():
    summarizer_model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)
    summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_model_name)

    keyword_model = KeyBERT(model="all-MiniLM-L6-v2")  # lightweight BERT for keywords

    return tokenizer, summarizer_model, keyword_model

tokenizer, summarizer_model, keyword_model = load_models()

# ------------------------
# Sidebar: Instructions
# ------------------------
with st.sidebar:
    st.image("Logo.png",
             width=120)
    st.title("InsightIQ 🔍")
    st.markdown("""
    Welcome to the **InsightIQ Text Summarizer !

    ✅ Paste long text or documents  
    ✅ Get a concise **abstractive summary**
    ---
    """)
    st.info("This tool uses state-of-the-art BART and BERT models under the hood.")

# ------------------------
# Main App
# ------------------------
st.title("📝 InsightIQ - Professional Text Summarizer")

# Text input
input_text = st.text_area(
    "📄 Paste your document or long text below:",
    height=300,
    placeholder="E.g., research abstract, article, report..."
)

# Options
col1, col2 = st.columns(2)
with col1:
    summarize_btn = st.button("🚀 Generate Summary")

# ------------------------
# Summarization
# ------------------------
if summarize_btn:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text to summarize.")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = summarizer_model.generate(
                inputs["input_ids"],
                num_beams=4,
                length_penalty=2.0,
                max_length=50,
                min_length=10,
                early_stopping=True,
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.success("✅ Summary generated!")
        st.write("### 🔹 Summary")
        st.write(summary)

# Footer
# ------------------------
st.markdown("""
<hr style='border-top: 1px solid #bbb;'>

<p style='text-align:center'>
    Developed with ❤️ by <strong>Intellivis.AI</strong> |
    Powered by <a href="https://huggingface.co/facebook/bart-large-cnn">BART</a> & <a href="https://github.com/MaartenGr/KeyBERT">KeyBERT</a>
</p>
""", unsafe_allow_html=True)
