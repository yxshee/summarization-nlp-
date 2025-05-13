from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TFAutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AdamWeightDecay,
    KerasMetricCallback,
    PreTrainedTokenizer
)
import evaluate
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import mixed_precision
from tensorflow.keras.losses import Loss

# **1. Enable Mixed Precision**
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# **2. Load the dataset**
config_name = 'punjabi'
dataset = load_dataset("csebuetnlp/xlsum", config_name)
print(dataset)

# **3. Display a sample from the dataset**
print(dataset['train'][0])

# **4. Initialize the tokenizer**
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# **5. Prefix for summarization tasks**
prefix = "summarize: "

# **6. Preprocessing function with further reduced sequence lengths**
def preprocess_fn(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)  # Reduced from 512 to 256

    targets = examples["summary"]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=32, truncation=True)  # Reduced from 64 to 32

    # Replace padding tokens with -100 to ignore them during loss calculation
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in l]
        for l in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]

    # **Create decoder_input_ids by shifting labels to the right and prepending the start token**
    decoder_start_token_id = tokenizer.pad_token_id  # T5 uses pad token id as decoder start
    decoder_input_ids = [
        [decoder_start_token_id] + l[:-1] if l else [decoder_start_token_id]
        for l in labels["input_ids"]
    ]

    model_inputs["decoder_input_ids"] = decoder_input_ids

    return model_inputs

# **7. Apply preprocessing to the dataset**
tokenized_dataset = dataset.map(preprocess_fn, batched=True, remove_columns=dataset["train"].column_names)
print(tokenized_dataset['train'][0])

# **8. Custom Data Collator for Seq2Seq**
from transformers import PreTrainedTokenizer

def custom_data_collator(features, tokenizer: PreTrainedTokenizer):
    # Get max length of input_ids
    max_length = max(len(feature['input_ids']) for feature in features)
    max_decoder_length = max(len(feature['decoder_input_ids']) for feature in features)
    
    # Initialize padded arrays
    batch_size = len(features)
    input_ids_padded = np.full((batch_size, max_length), tokenizer.pad_token_id)
    attention_mask_padded = np.zeros((batch_size, max_length))
    decoder_input_ids_padded = np.full((batch_size, max_decoder_length), tokenizer.pad_token_id)
    labels_padded = np.full((batch_size, max_decoder_length), -100)  # -100 is the ignore index

    # Fill in the arrays with actual values
    for i, feature in enumerate(features):
        input_length = len(feature['input_ids'])
        decoder_length = len(feature['decoder_input_ids'])
        
        input_ids_padded[i, :input_length] = feature['input_ids']
        attention_mask_padded[i, :input_length] = feature['attention_mask']
        decoder_input_ids_padded[i, :decoder_length] = feature['decoder_input_ids']
        labels_padded[i, :decoder_length] = feature['labels'][:decoder_length]

    # Convert to TensorFlow tensors
    return {
        "input_ids": tf.convert_to_tensor(input_ids_padded, dtype=tf.int64),
        "attention_mask": tf.convert_to_tensor(attention_mask_padded, dtype=tf.int64),
        "decoder_input_ids": tf.convert_to_tensor(decoder_input_ids_padded, dtype=tf.int64),
        "labels": tf.convert_to_tensor(labels_padded, dtype=tf.int64)
    }

# Add custom loss class
class MaskedSparseCategoricalCrossentropy(Loss):
    def __init__(self, from_logits=True, reduction='none', name='masked_sparse_categorical_crossentropy'):
        super().__init__(name=name, reduction=reduction)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # Create mask for -100 values
        mask = tf.not_equal(y_true, -100)
        # Replace -100 with 0 to avoid invalid values
        y_true_cleaned = tf.where(mask, y_true, 0)
        
        # Calculate regular sparse categorical crossentropy
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_cleaned, y_pred, from_logits=self.from_logits
        )
        
        # Apply mask and calculate mean over non-masked values
        mask = tf.cast(mask, loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

# Load and compile the model before dataset preparation
model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
optimizer = AdamWeightDecay(learning_rate=5e-5, weight_decay_rate=0.01)
model.compile(
    optimizer=optimizer,
    loss=MaskedSparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Define the data collator
datacollator = lambda x: custom_data_collator(x, tokenizer)

# Update batch size to help with memory issues
BATCH_SIZE = 2

# Update dataset preparation
train_set = model.prepare_tf_dataset(
    tokenized_dataset["train"],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=datacollator
)

val_set = model.prepare_tf_dataset(
    tokenized_dataset["validation"],
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=datacollator
)

test_set = model.prepare_tf_dataset(
    tokenized_dataset["test"],
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=datacollator
)

# **15. Initialize the Rouge evaluator**
rouge = evaluate.load("rouge")

# **16. Function to compute metrics**
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE metrics
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Calculate average generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# **17. Define a callback for metrics evaluation**
callback = KerasMetricCallback(
    metric_fn=compute_metrics,
    eval_dataset=val_set,
    predict_with_generate=True,
    use_xla_generation=True,
    generate_kwargs={"max_length": 32}  # Adjusted to match label max_length
)

# **18. Train the model**
model.fit(
    train_set,
    validation_data=val_set,
    epochs=2,
    callbacks=[callback]
)

# **19. Save the tokenizer**
tokenizer.save_pretrained("my_t5_summarization")
