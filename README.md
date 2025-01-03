

# Summarization NLP

This project is designed to generate concise and coherent summaries from extensive textual data. Leveraging advanced machine learning algorithms and state-of-the-art deep learning architectures, this project aims to facilitate efficient information digestion, enabling users to grasp key insights swiftly.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset Information](#dataset-information)
- [Model Architecture](#model-architecture)
  - [Chosen Models](#chosen-models)
  - [Training Strategy](#training-strategy)
- [Model Evaluation](#model-evaluation)
  - [Performance Metrics](#performance-metrics)
  - [Sample Summaries](#sample-summaries)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

In the era of information overload, the ability to distill vast amounts of text into succinct summaries is invaluable. **Summarization NLP** addresses this need by providing automated tools to generate high-quality summaries from diverse textual sources, including articles, reports, and social media content. Whether you're a researcher aiming to synthesize literature or a professional seeking quick insights, this project offers reliable and efficient summarization capabilities.

The **fine-tuned T5 model** used in this project is available on [Hugging Face](https://huggingface.co/yxshee/t5-transformer), making it easy to integrate into your NLP workflows.

---

## Features

- **Abstractive Summarization**: Generates novel sentences that capture the essence of the input text, mimicking human-like summaries.
- **Customizable Summary Length**: Allows users to specify the desired length of the summary.
- **API Integration**: Offers RESTful APIs for seamless integration into other applications and services.
- **User-Friendly Interface**: Intuitive web interface for easy access and usage.

---

## Dataset Information

### Source

The dataset used for fine-tuning this model is the **[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum)** dataset. XL-Sum is a multilingual summarization dataset that provides professionally written summaries for news articles across 44 languages.


#### DataFrame Overview

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 300000 entries, 0 to 299999
Data columns (total 3 columns):
 #   Column   Non-Null Count   Dtype 
---  ------   --------------   ----- 
 0   id       300000 non-null  int64 
 1   article  300000 non-null  object
 2   summary  300000 non-null  object
dtypes: int64(1), object(2)
memory usage: 6.8+ MB
```

### Sample Data

**First 5 Rows of the Dataset:**

| id     | article                                            | summary                            |
|--------|----------------------------------------------------|------------------------------------|
| 1      | The quick brown fox jumps over the lazy dog...     | Quick fox jumps over lazy dog.     |
| 2      | In recent news, the stock market has seen significant... | Stock market experiences significant changes. |
| 3      | Advances in artificial intelligence have paved the way... | AI advancements pave the way for future technologies. |
| 4      | The culinary world has been revolutionized by...    | Culinary world sees major changes. |
| 5      | Environmental concerns are at an all-time high as...| Environmental concerns rise sharply. |

---


---

## Model Architecture

### Chosen Models

This project employs the **T5 (Text-to-Text Transfer Transformer)**, a versatile model that treats every NLP problem as a text generation task. The T5 model has been fine-tuned on the XL-Sum dataset for generating high-quality abstractive summaries.

### Training Strategy

1. **Data Splitting**:
   - **Training Set**: 80%
   - **Validation Set**: 10%
   - **Testing Set**: 10%

2. **Optimization Algorithm**:
   - AdamW optimizer was used for efficient training.

3. **Frameworks**:
   - TensorFlow was used to train the model.

---

## Model Evaluation

### Performance Metrics

1. **ROUGE Scores**:
   - **ROUGE-1**: Measures the overlap of unigrams between the generated summary and the reference summary.
   - **ROUGE-2**: Measures the overlap of bigrams.
   - **ROUGE-L**: Measures the longest common subsequence between the generated summary and the reference summary.

### Sample Summaries

**Original Article:**

*"In recent news, the stock market has seen significant volatility due to geopolitical tensions. Investors are concerned about the potential impact on global trade and economic stability. Analysts suggest that diversification and cautious investment strategies are advisable in the current climate."*

**Generated Summary:**

*"Geopolitical tensions have caused significant volatility in the stock market, raising concerns about global trade and economic stability. Analysts recommend diversification and cautious investment strategies."*

---

## Installation

To set up the **Summarization NLP** project locally, follow the steps below:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yxshee/summarization-nlp.git
   cd summarization-nlp
   ```

2. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Fine-Tuned Model**:
   - The fine-tuned T5 model is available on [Hugging Face](https://huggingface.co/yxshee/t5-transformer). Download the model and tokenizer:
     ```python
     from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

     tokenizer = AutoTokenizer.from_pretrained("yxshee/t5-transformer")
     model = TFAutoModelForSeq2SeqLM.from_pretrained("yxshee/t5-transformer")
     ```

4. **Prepare the Dataset**:
   - If using custom datasets, update the configuration accordingly.

5. **Run Preprocessing Scripts**:
   ```bash
   python preprocess.py
   ```

6. **Train the Model**:
   ```bash
   python train.py
   ```

---

## Usage

### Generating Summaries

You can generate summaries using the fine-tuned T5 model.

**Example Using Python**:

```python
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yxshee/t5-transformer")
model = TFAutoModelForSeq2SeqLM.from_pretrained("yxshee/t5-transformer")

# Input text
text = "In recent news, the stock market has seen significant volatility due to geopolitical tensions..."

# Tokenize input
inputs = tokenizer("summarize: " + text, return_tensors="tf", max_length=512, truncation=True)

# Generate summary
outputs = model.generate(inputs["input_ids"], max_length=100, num_beams=4, early_stopping=True)

# Decode and print the summary
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Deployment

The fine-tuned T5 model can be deployed as a RESTful API or integrated into existing NLP pipelines. Refer to the detailed deployment instructions in the repository.

---

## Future Enhancements

1. **Multi-Language Support**: Extend summarization functionality to other languages.
2. **Real-Time Summarization**: Optimize the model for real-time summarization tasks.
3. **Interactive Web Interface**: Develop an enhanced web interface for batch processing and history tracking.

---

## Contributing

Contributions are welcome! Please refer to the contributing guidelines in the repository for more details.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software under the terms of the license.

---

## Acknowledgements

- **Hugging Face**: For providing the T5 model and the XL-Sum dataset.
- **TensorFlow**: For enabling efficient model training and deployment.

--- 
