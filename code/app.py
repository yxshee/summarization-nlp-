"""
T5 Summarization Web App
Streamlit-based interface for text summarization using T5.
"""

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Page configuration
st.set_page_config(
    page_title="T5 Summarizer",
    page_icon="📝",
    layout="centered",
)

# Constants
MODEL_CHECKPOINT = "t5-small"
PREFIX = "summarize: "


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_model():
    """Load the T5 model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
        device = get_device()
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def generate_summary(text: str, tokenizer, model, device, max_length: int, num_beams: int) -> str:
    """Generate summary for the given text."""
    full_input = PREFIX + text
    
    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    ).to(device)
    
    summary_ids = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def main():
    # Header
    st.image(
        "https://media.giphy.com/media/l2W5P0NgGeR3Iy61q4/giphy.gif",
        width=150,
    )
    st.title("📝 T5 Summarizer")
    st.write("Enter text and get a concise summary powered by T5!")
    
    # Load model
    with st.spinner("Loading model..."):
        tokenizer, model, device = load_model()
    
    if tokenizer is None or model is None:
        st.error("Failed to load model. Please check your internet connection.")
        st.stop()
    
    st.success(f"Model loaded on: {device}")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    max_length = st.sidebar.slider("Max summary length", 32, 256, 128)
    num_beams = st.sidebar.slider("Beam search width", 1, 8, 4)
    
    # Text input
    input_text = st.text_area(
        "Paste your text here:",
        height=200,
        placeholder="Enter the text you want to summarize...",
    )
    
    # Summarize button
    if st.button("🚀 Summarize", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    summary = generate_summary(
                        input_text, tokenizer, model, device, max_length, num_beams
                    )
                    st.subheader("📋 Summary")
                    st.write(summary)
                    
                    # Show stats
                    st.caption(
                        f"Input: {len(input_text.split())} words → "
                        f"Summary: {len(summary.split())} words"
                    )
                except Exception as e:
                    st.error(f"Error generating summary: {e}")


if __name__ == "__main__":
    main()
