# summarization-nlp
Used text summarization reduces a document to a shorter version, retaining key points. Extractive summarization selects important content from the source.
The XLSum dataset (Punjabi) includes articles and summaries, aiding NLP tool development for underrepresented languages.

_**NLP Application: Text Summarization**_

Text summarization is the process of reducing a text document to a shorter
version, providing a summary that retains the most important points of the
original document. This is particularly useful in an age where information
overload is a common challenge.

There are two primary types of **text summarization** :

1. **Extractive Summarization**: This method involves selecting important
sentences, paragraphs, or phrases directly from the source document to
create a summary. It does not generate new text but rather extracts and
compiles existing content.
2. **Abstractive Summarization**: This approach involves generating new
sentences that convey the most critical information from the original text. It
uses advanced machine learning techniques to interpret and paraphrase
the content, often resulting in more coherent and concise summaries
compared to extractive methods.

_**Dataset: XLSum (Punjabi Language)**_

The "csebuetnlp/xlsum" dataset is a comprehensive dataset containing
multilingual news articles and their summaries. The dataset includes a wide
range of languages and provides a valuable resource for training and evaluating
text summarization models.For this project, the XLSum dataset from the data
library is used, specifically focusing on the Punjabi language.

Details of the XLSum Dataset:

● Articles: The dataset contains articles written in the Punjabi language,
covering a variety of topics from news to general information.

● Summaries: Each article is paired with a summary that captures the
essence of the content. These summaries are used as ground truth for
training and evaluating the summarization model.
● Language Focus: The use of Punjabi language data is particularly
significant for developing NLP tools that support underrepresented
languages, enhancing the accessibility and utility of such tools.
The dataset provides a valuable resource for training and evaluating
summarization models in the Punjabi language, contributing to the broader goal
of supporting NLP text summarization.

_**Transformer Model: T5(Text-To-Text Transfer Transformer)**_

In this project, we utilized the T5 model, specifically the t5-small checkpoint, to
perform text summarization. T5, developed by Google Research, is a versatile
transformer model designed to handle a wide range of NLP tasks by converting
all tasks into a text-to-text format.

**Model Architecture and Features:**

● Model Name: T5 (Text-To-Text Transfer Transformer)
● Checkpoint: t5-small
● Parameters: Approximately 60 million parameters
● Architecture: Encoder-Decoder architecture, suitable for sequence-to-
sequence tasks
● Preprocessing: The input texts are prefixed with "summarize: " before
tokenization to indicate the task.

**Tokenization and Preprocessing:**

The **AutoTokenizer** from the Hugging Face transformers library is used for
tokenization. The preprocessing function prepares the input texts by adding a
"summarize: " prefix and tokenizing both the inputs and the target summaries.

**Training Setup:**

1. Optimizer: AdamWeightDecay with a learning rate of 2e-5 and weight decay
of 0.01.

2. Data Collation: Using DataCollatorForSeq2Seq to dynamically pad inputs
and outputs during batching.

4. Training Data: The tokenized dataset is divided into training, validation, and
test sets.

5. Batch Size: 8 for both training and evaluation.

6. Evaluation Metric: ROUGE (Recall-Oriented Understudy for Gisting
Evaluation) is used to evaluate the summarization quality.

**Optimizer: AdamWeightDecay**

To train our model effectively, we used the AdamWeightDecay optimizer. This
optimizer is a variant of the Adam optimizer, incorporating weight decay to
prevent overfitting. It helps in maintaining a balance between model accuracy
and generalization.

**Model Training:**

The model is compiled and trained using TensorFlow, with the optimizer and data
collator set up as described. Training involves feeding the tokenized dataset into
the model, computing loss, and updating the model's weights.

**Evaluation:**

The model's performance is evaluated using the ROUGE metric, which measures
the overlap between the generated summaries and the reference summaries.

_**Results:**_

After training the model for 3 epochs, we evaluate its performance using
the ROUGE metric.

The results are as follows:

● ROUGE-1 Score: (value)

● ROUGE-2 Score: (value)

● ROUGE-3 Score: (value)

These scores indicate the model's ability to generate summaries that are
similar to the reference summaries in terms of n-gram overlay.
