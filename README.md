\# 🛒 Text Classification on E-commerce Data using FastText



\## 📌 Project Overview

This project implements \*\*text classification on an e-commerce dataset using Facebook’s FastText library\*\*. The aim is to efficiently classify product-related text into predefined categories while maintaining high accuracy and fast training time. FastText’s use of word and subword embeddings makes the model robust to unseen and misspelled words, which are common in real-world e-commerce data.



---



\## ⚙️ Tech Stack

\- Python  

\- FastText  

\- Pandas, NumPy  

\- Natural Language Processing (NLP)  

\- Jupyter Notebook  



---



\## 🧠 Problem Statement

E-commerce platforms generate massive amounts of textual data such as product titles, descriptions, and reviews. Traditional machine learning models can be slow and resource-intensive. This project explores \*\*FastText as a scalable and lightweight NLP solution\*\* for accurate multi-class text classification.



---



\## 🛠️ Implementation Steps

1\. Loaded and explored the e-commerce text dataset.

2\. Preprocessed text data (cleaning, tokenization, label formatting).

3\. Converted data into FastText-compatible supervised format.

4\. Trained a FastText classification model using word and subword embeddings.

5\. Evaluated the model on unseen test samples.

6\. Analyzed and visualized prediction outputs.



---



\## 📊 Results \& Observations

\- Achieved \*\*strong classification performance\*\* with \*\*very low training time\*\*.

\- Subword embeddings improved robustness to \*\*out-of-vocabulary and misspelled words\*\*.

\- The model generalized well on unseen e-commerce text samples.

\- Demonstrates suitability for \*\*real-time and large-scale NLP applications\*\*.



---



\## 📷 Output GIF





\### 🔹 Prediction Results

!\[Prediction Output](output/op.gif)



> 📌 Output images are stored inside the `output/` directory.



---



\## 📁 Project Structure



Text-Classification-FastText/

│

├── Text\_classification\_fasttext\_ecommerce\_dataset.ipynb

├── README.md

├── requirements.txt

├── dataset/ # optional

└── outputs/

├── training\_output.png

└── prediction\_output.png







---



\## 🚀 Key Learnings

\- Practical experience with \*\*FastText-based NLP classification\*\*

\- Built an end-to-end \*\*text preprocessing and training pipeline\*\*

\- Understood trade-offs between speed, accuracy, and scalability

\- Hands-on exposure to \*\*production-relevant NLP techniques\*\*



---



\## 🔮 Future Improvements

\- Compare performance with TF-IDF + Logistic Regression

\- Tune FastText hyperparameters

\- Deploy the model as a REST API



---



\## ▶️ How to Run

```bash

pip install -r requirements.txt





