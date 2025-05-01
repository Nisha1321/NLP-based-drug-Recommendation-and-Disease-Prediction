# NLP-based-drug-Recommendation-and-Disease-Prediction
# 🧠 Disease Condition Detection from Drug Reviews using NLP & Machine Learning

## 📌 Problem Statement

With the internet flooded by user-generated reviews on drug efficacy and side effects, there's a goldmine of healthcare insights hidden in plain sight. Manually analyzing these reviews at scale is not just inefficient — it's practically impossible. This project tackles that challenge by building a machine learning model that **automatically predicts disease conditions based on user-submitted drug reviews**.

> By automating this process, we aim to support healthcare providers and patients in making more informed, data-backed decisions.

---

## 🚀 Project Summary & Introduction

This project harnesses the power of **Natural Language Processing (NLP)** and **machine learning** to classify disease conditions from textual drug reviews. Key steps include:
- Text preprocessing & feature extraction
- Training & evaluating classification models
- Generating predictions from new reviews

We use classic ML models — **Naive Bayes** and **Passive Aggressive Classifier** — achieving impressive performance, particularly on high-frequency conditions.

---

## 🎯 Aim & Objectives

**Aim:**  
To develop an ML-based system that predicts the disease condition from a user-submitted drug review.

**Objectives:**
- 📄 Preprocess and clean text data effectively.
- 🔍 Extract meaningful numerical features from text.
- 🤖 Train and evaluate ML models for classification.
- 🛠️ Build an extendable system for future data and models.

---

## 🧩 Design: Analysis & Methodology

### 1. 📦 Program / Module Specifications

| Module                 | Description |
|------------------------|-------------|
| Data Preprocessing     | Cleans text: removes noise, stopwords, lemmatizes. |
| Feature Extraction     | Converts text to numerical data using BoW & TF-IDF. |
| Model Training         | Trains classifiers (Naive Bayes & Passive Aggressive). |
| Evaluation             | Analyzes accuracy and confusion matrices. |

---

### 2. 🏗️ Model Construction

- **Text Preprocessing:**
  - HTML tag removal
  - Lowercasing, punctuation & number removal
  - Stopword removal
  - Lemmatization

- **Feature Extraction:**
  - Bag of Words (BoW)
  - Term Frequency-Inverse Document Frequency (TF-IDF)

- **Model Selection:**
  - **Naive Bayes Classifier** – High performance on text classification
  - **Passive Aggressive Classifier** – Real-time, adaptable, robust

- **Evaluation:**
  - Accuracy Score
  - Confusion Matrix

---

## 📚 Dataset Details

**Source:** Drug reviews dataset with metadata including condition, rating, and review date.

### Key Fields:
- `Drug Name`
- `Condition`
- `Review Content`
- `Rating`
- `Date`
- `Useful Count`

### Preprocessing Steps:
- Text cleaning (removing special chars, excess whitespace)
- Tokenization
- Padding
- Data augmentation (synonym replacement/paraphrasing)

### Top Categories:
| Condition         | No. of Reviews |
|-------------------|----------------|
| Birth Control     | 28,788         |
| Depression        | 9,069          |
| Pain              | 5,800          |
| Arthritis         | 6,500          |
| Anxiety           | 4,000          |
| Cholesterol       | 3,400          |
| Asthma            | 2,700          |
| Acid Reflux       | 2,200          |
| High Blood Pressure| 2,321         |
| Diabetes          | 1,500          |

---

## 💻 Implementation Overview

**Modules Implemented:**
- 📥 `Data Loading Module` – Reads dataset
- 🧹 `Preprocessing Module` – Cleans and tokenizes text
- 🧠 `Feature Extraction Module` – Uses BoW & TF-IDF
- 🏋️ `Model Training Module` – Naive Bayes and Passive Aggressive Classifier
- 📊 `Evaluation Module` – Computes accuracy & confusion matrix

---

## ✅ Results

- **Naive Bayes Classifier**:
  - 🔥 **Accuracy:** 92.3%
  - ⚡ Excellent on frequent conditions (e.g., Birth Control, Depression)
  - ⚠️ Moderate performance on low-frequency conditions

- **Passive Aggressive Classifier**:
  - Solid backup option
  - Best suited for real-time systems

---

## 📈 Evaluation Metrics

| Metric           | Description |
|------------------|-------------|
| **Accuracy**     | Overall performance of the model |
| **Confusion Matrix** | Breakdown of correct vs incorrect predictions per condition |

---

## 🧠 Conclusion

This project proves the potential of **data-driven healthcare** through NLP and ML. By classifying disease conditions from textual reviews, the system:
- Streamlines insights from unstructured data
- Performs exceptionally well for common conditions
- Demonstrates scalability for future integrations

---

## 🔮 Future Work

- 📈 **Expand Dataset:** Add more reviews for underrepresented conditions
- 🧠 **Advanced Models:** Try deep learning (LSTM, BERT) for complex patterns
- 🔁 **Feedback Loop:** Let users validate or correct predictions

---

## ⚠️ Limitations

- **Data Imbalance:** Skewed review counts across conditions
- **Limited Condition Set:** Some conditions aren't represented
- **Complex Texts:** Multi-condition reviews challenge simple classifiers

---

## 🙌 Acknowledgments

Thanks to the open-source and data science community for providing resources, frameworks, and datasets that made this project possible.

---

## 🛠️ Technologies Used

- Python 🐍
- Scikit-learn 🤖
- NLTK / spaCy 🧹
- Pandas & NumPy 📊
- Matplotlib & Seaborn 📈

---


