# Spam-Filter

#  SMS Spam Filter — Machine Learning Project

This project implements a complete end-to-end **SMS spam detection system** using classical NLP techniques and supervised machine learning.  
It was developed as part of a data science case study and follows industry-standard methodology (CRISP-DM).

The goal is to support the service team of a company that plans to open a public communication channel, requiring a **reliable, adjustable, and production-ready spam-filtering model**.

---


---

##  Business Objective

The company plans to open a public messaging channel for customer interaction.  
However, this exposes the platform to **spam**, which can:

- undermine customer trust  
- overload the service team  
- introduce harmful or irrelevant content  

The system must:

- detect spam messages automatically  
- support **adjustable risk levels** (low-risk, default, high-risk)  
- minimise false negatives (spam classified as ham)  
- provide a safe, transparent, and easily deployable workflow  

---

##  Dataset

The project uses the **SMS Spam Collection Dataset**, containing:

- **5,574** English SMS messages  
- Each labeled as **ham** (legitimate) or **spam**  
- Collected from multiple public sources → highly diverse writing styles  

Challenges:

- informal text  
- inconsistent grammar and slang  
- class imbalance (ham ≫ spam)  

These factors influenced preprocessing choices, model development, and threshold tuning.

---

##  Data Preparation

Main preprocessing steps:

- Lowercasing  
- Removing URLs, emails, digits, and punctuation  
- Removing duplicates  
- Tokenization  
- Lemmatization  
- Stopword removal  
- Final cleaned feature used for TF-IDF

This standardized the text and improved model interpretability and robustness.

---

##  Model Development

Two supervised learning models were evaluated:

### **1. Naive Bayes (Baseline)**
- Very high ham recall (1.00)  
- Fast and simple  
- But low spam recall (0.79)  
- High false-negative rate → unsafe for business

### **2. Logistic Regression (Final Model)**
- Higher spam recall (0.91)  
- Lower false-negatives  
- Better separation of borderline cases  
- Calibrated probabilities → perfect for threshold tuning  

**Final choice: Logistic Regression**

---

##  Adjustable Spam-Risk Levels (Threshold Tuning)

The classifier supports **three modes**, based on probability thresholds:

| Mode        | Threshold | Behaviour |
|-------------|-----------|-----------|
| Low-risk    | 0.30      | Very strict, minimizes undetected spam but increases false positives |
| Default     | 0.50      | Best overall balance (recommended) |
| High-risk   | 0.80      | Very permissive, protects customer experience but lets more spam through |

This flexibility allows the service team to adapt filtering based on:

- spam waves  
- marketing periods  
- customer tolerance  
- operational workload  

---

##  Error Analysis

Error inspection was performed for FP and FN in each threshold mode, revealing:

- informal ham messages sometimes resemble spam  
- short spam messages can resemble casual conversational text  
- borderline cases justify having adjustable threshold modes  

This step increases transparency and builds trust in the system.

---

##  Deployment Readiness

### Saved Artifacts
- TF-IDF vectorizer (`tfidf_vectorizer.joblib`)
- Logistic Regression model (`logreg_spam_classifier.joblib`)

### Production-Ready Inference Function
`classify_message()`:
- applies preprocessing  
- transforms text via TF-IDF  
- produces **P(spam)**  
- applies selected risk mode  
- returns label + probability  

This simulates how the system would run in real messaging workflows, and can be integrated into:

- internal tools  
- APIs  
- a GUI  
- automated message routing systems  

---

##  Example Prediction

```python
classify_message("You won a prize! Call now!", mode="low")

{
  "clean_text": "...",
  "probability_spam": 0.9842,
  "label": "spam",
  "threshold_used": 0.30,
  "mode": "low"
}



