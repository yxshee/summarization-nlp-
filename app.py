import streamlit as st
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# 1. Title and basic instructions for the user.
st.title("T5 Summarization App")
st.write("Enter text in Punjabi (or any other language if your model supports it), and get a summarized output!")

# 2. Function to load model and tokenizer.
@st.cache_resource
def load_model_and_tokenizer(model_checkpoint: str = "my_t5_summarization/tf_model.h5"):
    """
    Loads the fine-tuned T5 model and tokenizer from a local directory or Hugging Face Hub.
    Replace 'my_t5_summarization' with the path to your model files
    or a Hugging Face Hub model ID (e.g., 'your-username/your-model-name').
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return tokenizer, model

# 3. Load your previously fine-tuned T5 model (and tokenizer).
#    If you pushed your model to Hugging Face hub, e.g. "your-username/your-model-id",
#    replace "my_t5_summarization" with that hub path.
tokenizer, model = load_model_and_tokenizer("my_t5_summarization")

# 4. Text input for the user to paste the content that needs summarizing.
input_text = st.text_area("Paste your text here:", height=200)

# 5. Summarize button
if st.button("Summarize"):
    if len(input_text.strip()) == 0:
        st.warning("Please enter some text to summarize.")
    else:
        # 6. Prepend the prefix "summarize: " for T5-based summarization.
        prefix = "summarize: "
        full_input = prefix + input_text

        # 7. Tokenize
        inputs = tokenizer(
            full_input,
            return_tensors="tf",
            max_length=1024,
            truncation=True
        )

        # 8. Generate summary (customize generation parameters as you see fit).
        summary_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            no_repeat_ngram_size=2
        )

        # 9. Decode summary output
        summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # 10. Display the summary
        st.subheader("Summary")
        st.write(summarized_text)
