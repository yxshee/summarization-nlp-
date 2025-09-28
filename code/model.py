"""
T5 Summarization Training Script (PyTorch)
Fine-tune T5-small on XL-Sum dataset for text summarization.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate
import numpy as np

# Configuration
CHECKPOINT = "t5-small"
DATASET_CONFIG = "punjabi"
OUTPUT_DIR = "./t5_summarization_model"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def load_data():
    """Load the XL-Sum dataset."""
    dataset = load_dataset("csebuetnlp/xlsum", DATASET_CONFIG)
    print(f"Dataset loaded: {dataset}")
    return dataset


def preprocess_function(examples, tokenizer):
    """Tokenize inputs and targets for summarization."""
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["text"]]
    
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )
    
    labels = tokenizer(
        examples["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer, rouge):
    """Compute ROUGE metrics for evaluation."""
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 with pad token for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    
    # Add generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def main():
    """Main training function."""
    print("=" * 50)
    print("T5 Summarization Training")
    print("=" * 50)
    
    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
    model.to(device)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_data()
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Load ROUGE metric
    rouge = evaluate.load("rouge")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # Enable FP16 on CUDA
        push_to_hub=False,
        report_to="none",
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, rouge),
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model and tokenizer
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate(tokenized_dataset["test"])
    print(f"\nTest Results: {results}")
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
