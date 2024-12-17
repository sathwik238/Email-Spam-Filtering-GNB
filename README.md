# Email Spam Filtering using Multinomial Naive Bayes and Transfer Learning

This project predicts whether an email is **spam** or **ham** (not spam). Initially, a **Multinomial Naive Bayes** model was used. To further improve performance, **Transfer Learning** is applied using pre-trained models like **BERT** or other deep learning architectures for text classification.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Model Workflow](#model-workflow)
5. [Transfer Learning Approach](#transfer-learning-approach)
6. [Results](#results)
7. [How to Run the Project](#how-to-run-the-project)
8. [Future Work](#future-work)
9. [References](#references)

---

## **Project Overview**

Spam detection is a classic text classification problem where emails are categorized as either **spam** (unwanted) or **ham** (legitimate). The project workflow includes:
- **Multinomial Naive Bayes** for baseline performance.
- Integration of **Transfer Learning** using **pre-trained models** like BERT for improved accuracy and robustness.

---

## **Dataset**

- **File**: `spam.csv`
- **Description**:
    - The dataset contains email samples labeled as:
        - `spam`: Unwanted or promotional emails.
        - `ham`: Non-spam, legitimate emails.
    - Features include email text content.

**Sample Data**:
| Label | Message                                           |
|-------|--------------------------------------------------|
| spam  | Free entry in 2 a wkly comp to win FA Cup final...|
| ham   | Nah I don't think he goes to usf anymore         |
| spam  | WINNER!! As a valued network customer you have...|
| ham   | Even my brother is not like to speak with me...  |

---

## **Technologies Used**

- **Programming Language**: Python
- **Libraries**:
    - **Baseline**:
        - pandas, scikit-learn, nltk, matplotlib, seaborn
    - **Transfer Learning**:
        - Hugging Face Transformers (`transformers` library)
        - PyTorch or TensorFlow/Keras for deep learning

---

## **Model Workflow**

1. **Data Preprocessing**:
    - Load and clean the dataset.
    - Text preprocessing:
        - Lowercasing, stopword removal, punctuation removal, tokenization.

2. **Baseline Model**:  
    - Train a **Multinomial Naive Bayes** model using:
        - Bag-of-Words (Count Vectorizer)
        - TF-IDF Vectorizer

3. **Transfer Learning Approach**:
    - Utilize **pre-trained models** like BERT or DistilBERT for text classification.
    - Fine-tune the pre-trained models on the email dataset for spam detection.

---

## **Transfer Learning Approach**

1. **Pre-Trained Model**:
   - Use a Transformer-based architecture (e.g., BERT or DistilBERT) from Hugging Face:
     ```python
     from transformers import BertTokenizer, BertForSequenceClassification
     from transformers import Trainer, TrainingArguments

     # Load Pre-Trained BERT Model
     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
     ```

2. **Tokenization**:
   Convert email text into tokens that the model can process:
   ```python
   encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors='pt')
