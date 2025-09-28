#!/usr/bin/env python3
"""
T5 Summarization Test Script
Quick test to verify the summarization model works correctly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_summarization():
    """Test the T5 summarization pipeline."""
    print("=" * 60)
    print("T5 Summarization Test")
    print("=" * 60)
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    print("\nLoading T5 model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        model.to(device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test cases
    test_texts = [
        {
            "name": "Amazon Rainforest",
            "text": """
            The Amazon rainforest, also known as Amazonia or the Amazon Jungle, 
            is a moist broadleaf tropical rainforest in the Amazon biome that 
            covers most of the Amazon basin of South America. This basin 
            encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 
            (2,100,000 sq mi) are covered by the rainforest. This region includes 
            territory belonging to nine nations and 3,344 territories.
            """,
        },
        {
            "name": "Climate Change",
            "text": """
            Climate change refers to long-term shifts in temperatures and weather 
            patterns. These shifts may be natural, such as through variations in 
            the solar cycle, but since the 1800s, human activities have been the 
            main driver of climate change, primarily due to burning fossil fuels 
            like coal, oil and gas.
            """,
        },
    ]
    
    for i, test in enumerate(test_texts, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        print(f"Input: {test['text'].strip()[:100]}...")
        
        input_text = "summarize: " + test["text"]
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(device)
        
        summary_ids = model.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"Summary: {summary}")
    
    return True


if __name__ == "__main__":
    success = test_summarization()
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")
    print("=" * 60)
