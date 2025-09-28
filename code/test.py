#!/usr/bin/env python3
"""
Simple test script for T5 summarization
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_summarization():
    print("Loading T5 model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        print("Model loaded successfully!")
        
        # Test text
        text = """
        The Amazon rainforest, also known as Amazonia or the Amazon Jungle, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 territories.
        """
        
        print("Input text:")
        print(text.strip())
        
        # Add T5 prefix
        input_text = "summarize: " + text
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        print("\nGenerating summary...")
        
        # Generate summary
        summary_ids = model.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        print("\nGenerated Summary:")
        print(summary)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_summarization()
    if success:
        print("\n✅ Summarization test completed successfully!")
    else:
        print("\n❌ Summarization test failed!")
