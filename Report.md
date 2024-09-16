
### 1. **Project Overview:**
The goal of this project is to build a **text summarization model** using an **extractive** approach. Text summarization reduces a document to a shorter version, retaining its key points. This is particularly useful in dealing with large amounts of information, making it easier to digest the main points quickly. 

The project leverages the **XLSum dataset**, focusing on the **Punjabi language**. This project is noteworthy for supporting an underrepresented language in Natural Language Processing (NLP), helping to improve accessibility for less commonly used languages.

### 2. **Technical Stack:**
- **Model Architecture**: The project uses **T5 (Text-to-Text Transfer Transformer)**, particularly the **t5-small** version, which is a pre-trained transformer model by Google for various NLP tasks. It uses an **encoder-decoder** architecture, ideal for sequence-to-sequence tasks like summarization.
- **Tokenizer**: Hugging Face's **AutoTokenizer** is utilized to prepare the data by prefixing the text inputs with `"summarize: "` before tokenization.
- **Optimizer**: The project uses the **AdamWeightDecay** optimizer, with a learning rate of `2e-5` and a weight decay of `0.01`, to prevent overfitting and help generalization.
- **Evaluation Metric**: The model's performance is evaluated using the **ROUGE** metric, which compares the overlap of n-grams between the generated and reference summaries.

### 3. **Python Code Breakdown**:

The project includes several Python scripts and Jupyter notebooks. Below is a step-by-step explanation of the **summarization.ipynb** file.

- **Data Loading**: The script begins by loading the **XLSum dataset**, specifically focused on the Punjabi language.
  
- **Preprocessing**: The input text is preprocessed using the **AutoTokenizer** from Hugging Face. This includes adding the `"summarize: "` prefix to indicate the task and then tokenizing both the input documents and the target summaries.
  
- **Model Setup**: The T5-small model is loaded, which has about **60 million parameters**. This model is fine-tuned for text summarization tasks.
  
- **Training**: The model is trained using a small **batch size of 8** due to memory constraints. The training dataset is divided into **training, validation, and test sets**.
  
- **Evaluation**: The script evaluates the model's performance after training using the **ROUGE-1, ROUGE-2, and ROUGE-L** scores, which measure the similarity between the generated and target summaries in terms of word or n-gram overlap.
  
- **Results**: After a few epochs (typically 3), the results are reported in terms of ROUGE scores, indicating the model's summarization capability.

### 4. **Key Metrics:**
- **ROUGE**: Used to measure the quality of summaries by calculating the overlap of n-grams (e.g., bigrams, trigrams) between the generated summaries and the reference summaries.
- **Training Epochs**: The model is typically trained for **3 epochs**, with the performance improving incrementally.
- **Batch Size**: The batch size used for training and evaluation is 8, which is common in transformer-based models to handle memory usage.

You can explore the code and dataset further from the repository [here](https://github.com/yxshee/summarization-nlp).

### Detailed Breakdown of Key Aspects of the NLP Summarization Project

### 1. **T5 Model (Text-to-Text Transfer Transformer)**

The **T5 model** was introduced by Google Research in the paper *"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"*. It is unique in the sense that it treats every NLP problem, from translation to summarization, as a **text-to-text task**. This means the input is text, and the output is also text. In the context of this project:

- **Encoder-Decoder Architecture**: The model uses a classic encoder-decoder setup, where:
  - The **encoder** reads and processes the input text, turning it into a sequence of vectors that represent the textual meaning.
  - The **decoder** then takes this representation and generates the output, which is a summary in this case.

By treating summarization as a text generation task, the model can be applied to a wide range of other tasks with only small adjustments in the prompt. In this project, the T5 model is pre-trained and then fine-tuned on the summarization task.

### 2. **Dataset: XLSum (Punjabi Subset)**

The **XLSum dataset** used in the project contains multilingual text summarization data, supporting over 40 languages. Here, the focus is on **Punjabi**, an underrepresented language in NLP. 

- **Training Process**: The dataset is split into **training**, **validation**, and **test sets**, which helps evaluate how well the model generalizes.
- The project tokenizes the input text (articles) and the target summaries (actual summaries from the dataset) before feeding it into the T5 model.

### 3. **Tokenization Using AutoTokenizer**

In NLP, **tokenization** is the process of breaking down text into smaller units (tokens) that the model can process. This project uses Hugging Face’s **AutoTokenizer**, which automatically selects the right tokenizer for the T5 model.

- The text input is prefixed with `"summarize: "` to help the model understand the task at hand.
- Tokenization ensures that the input sequences are converted into a format the model can handle, specifically by turning them into token IDs.

Example code from the notebook:
```python
tokenizer = AutoTokenizer.from_pretrained('t5-small')
input_text = "summarize: " + article
input_ids = tokenizer(input_text, return_tensors='pt').input_ids
```

This process converts text into a list of numerical tokens that correspond to words or subwords, which the model uses for training and inference.

### 4. **Training Setup and Optimization**

- **AdamW Optimizer**: The model is trained using the **AdamW** optimizer. This optimizer is a variant of Adam but includes **weight decay**, which helps to prevent overfitting by adding a regularization term to the loss function.
  - **Learning Rate**: Set to `2e-5`, a small learning rate is chosen to ensure the model converges slowly and avoids missing important features in the data.
  - **Batch Size**: The project uses a batch size of 8, meaning 8 text summaries are processed at once during training. This is a common practice to prevent memory overload when using large transformer models.

```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

### 5. **Training and Fine-Tuning the Model**

The model is fine-tuned for the task of summarization over **multiple epochs** (typically 3 epochs in this case). Fine-tuning involves updating the pre-trained model weights slightly to better match the target task—in this case, summarizing Punjabi articles.

- During each epoch, the model processes all the training examples, updating its weights to minimize the difference between its generated summaries and the actual summaries.
- **Loss Function**: The loss is computed using **cross-entropy**, which is common in sequence prediction tasks like summarization.

### 6. **Evaluation: ROUGE Metrics**

To measure how well the model performs, the project uses **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation), a set of metrics commonly used for evaluating summaries. ROUGE measures the overlap between the generated summary and a reference summary in terms of n-grams, which are contiguous sequences of words.

- **ROUGE-1**: Measures the overlap of unigrams (single words) between the generated summary and the reference summary.
- **ROUGE-2**: Measures the overlap of bigrams (two-word pairs).
- **ROUGE-L**: Measures the longest common subsequence between the generated and reference summaries.

These metrics give a good indication of the model’s performance:
- **Higher scores** mean that the generated summary is more similar to the reference summary.

```python
# Example evaluation using the rouge metric
rouge_score = rouge_scorer.compute(predictions=generated_summaries, references=reference_summaries)
```

### 7. **Fine-Tuning Hyperparameters**

The model’s performance is highly dependent on hyperparameters like:
- **Learning Rate**: A small learning rate is essential for transformer models to ensure the model doesn’t overshoot during optimization.
- **Weight Decay**: A regularization technique to prevent overfitting, keeping the model generalizable to unseen data.
- **Batch Size**: Balancing the batch size is crucial as large batches may lead to faster training but risk memory overflow, while smaller batches can slow down the process but handle memory constraints better.

The default settings (e.g., `2e-5` learning rate, `0.01` weight decay, and 8 batch size) are commonly used starting points for fine-tuning transformer models.

---

### Summary of Project Workflow:
1. **Data Preprocessing**: Tokenization and text preparation using AutoTokenizer.
2. **Model Setup**: Loading the T5-small pre-trained model.
3. **Training**: Fine-tuning the model on the XLSum Punjabi dataset.
4. **Evaluation**: Assessing model performance with ROUGE metrics.
5. **Inference**: Using the trained model to generate summaries for new text inputs.

This workflow ensures the T5 model is capable of producing concise, relevant summaries, particularly for text in the Punjabi language.

---
