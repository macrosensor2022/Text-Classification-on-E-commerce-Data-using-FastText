# E-commerce Text Classification using FastText

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![FastText](https://img.shields.io/badge/FastText-0.9.2-orange.svg)](https://fasttext.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A scalable NLP text classification system leveraging Facebook's FastText library for efficient categorization of e-commerce product data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project implements an efficient text classification pipeline for e-commerce product data using **Facebook's FastText** library. The system classifies product titles, descriptions, and reviews into predefined categories with high accuracy and minimal training time. FastText's subword embedding approach ensures robustness to typos, misspellings, and out-of-vocabulary words commonly found in real-world e-commerce data.

### Why FastText?

- **Speed**: Trains significantly faster than traditional deep learning models
- **Subword Embeddings**: Handles rare words and typos effectively
- **Scalability**: Suitable for large-scale production environments
- **Efficiency**: Low memory footprint and fast inference

---

## 🔍 Problem Statement

E-commerce platforms generate vast amounts of unstructured textual data daily, including:
- Product titles and descriptions
- Customer reviews and feedback
- Search queries and recommendations

Traditional machine learning models often struggle with:
- Long training times on large datasets
- Poor handling of misspelled or rare words
- High computational resource requirements

This project addresses these challenges by implementing FastText, a lightweight yet powerful NLP solution optimized for text classification tasks.

---

## ✨ Key Features

- **Multi-class Text Classification**: Categorizes products into multiple predefined classes
- **Subword N-gram Embeddings**: Captures morphological patterns for better generalization
- **Fast Training & Inference**: Optimized for real-time applications
- **Robust to Noise**: Handles typos, abbreviations, and informal language
- **End-to-End Pipeline**: From data preprocessing to model evaluation
- **Jupyter Notebook Integration**: Interactive exploration and experimentation

---

## 🏗️ Technical Architecture

### Pipeline Workflow

```
Raw E-commerce Text
        ↓
Data Preprocessing (Cleaning, Tokenization)
        ↓
FastText Format Conversion (Supervised Learning)
        ↓
Model Training (Word + Subword Embeddings)
        ↓
Evaluation & Validation
        ↓
Predictions on Unseen Data
```

### Model Components

1. **Text Preprocessing**
   - Lowercasing and punctuation removal
   - Special character handling
   - Label formatting for FastText compatibility

2. **FastText Model**
   - Architecture: Bag of Words + Character N-grams
   - Embedding Dimension: Configurable (default: 100)
   - Learning Rate: Adaptive with tuning
   - Loss Function: Softmax for multi-class classification

3. **Evaluation Metrics**
   - Precision, Recall, F1-Score
   - Confusion Matrix Analysis
   - Per-class Performance Breakdown

---

## 📊 Dataset

- **Source**: E-commerce product text dataset
- **Format**: CSV with product descriptions and category labels
- **Size**: [Specify dataset size if available]
- **Categories**: Multiple product categories (e.g., Electronics, Fashion, Home & Kitchen, etc.)

### Data Structure

| Column | Description |
|--------|-------------|
| `text` | Product title/description |
| `label` | Product category |

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/macrosensor2022/Text-Classification-on-E-commerce-Data-using-FastText.git
cd Text-Classification-on-E-commerce-Data-using-FastText
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Required Libraries

```
fasttext==0.9.2
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 💻 Usage

### Training the Model

1. **Prepare your data** in FastText format:
```
__label__<category> <text>
```

2. **Run the Jupyter Notebook**:
```bash
jupyter notebook Text_classification_fasttext_ecommerce_dataset.ipynb
```

3. **Or use the Python script**:
```python
import fasttext

# Train model
model = fasttext.train_supervised(
    input="train.txt",
    lr=0.1,
    epoch=25,
    wordNgrams=2,
    dim=100
)

# Evaluate
results = model.test("test.txt")
print(f"Precision: {results[1]:.4f}")
print(f"Recall: {results[2]:.4f}")
```

### Making Predictions

```python
# Load trained model
model = fasttext.load_model("model.bin")

# Predict category
text = "wireless bluetooth headphones with noise cancellation"
prediction = model.predict(text)
print(f"Category: {prediction[0][0]}")
print(f"Confidence: {prediction[1][0]:.4f}")
```

---

## 📈 Results

### Model Performance

- **Training Time**: Significantly faster compared to LSTM/BERT models
- **Accuracy**: High classification accuracy on test set
- **Robustness**: Excellent handling of misspelled and rare words
- **Inference Speed**: Real-time predictions suitable for production deployment

### Key Observations

✅ **Fast Training**: Model converges quickly even on large datasets  
✅ **Subword Advantage**: Effectively handles typos like "wireles headphons"  
✅ **Generalization**: Strong performance on unseen product descriptions  
✅ **Scalability**: Suitable for large-scale e-commerce applications  

### Visualization

Prediction outputs and performance metrics are available in the `output/` directory:
- Training/validation curves
- Confusion matrices
- Per-category performance analysis

---

## 📁 Project Structure

```
Text-Classification-on-E-commerce-Data-using-FastText/
│
├── Text_classification_fasttext_ecommerce_dataset.ipynb  # Main notebook
├── README.md                                               # Project documentation
├── requirements.txt                                        # Python dependencies
├── .gitignore                                              # Git ignore rules
│
├── dataset/                                                # Training data (optional)
│   ├── train.txt
│   └── test.txt
│
└── output/                                                 # Results and visualizations
    ├── op.gif                                              # Demo output
    ├── training_output.png
    └── prediction_output.png
```

---

## 🔮 Future Enhancements

### Planned Improvements

- [ ] **Hyperparameter Tuning**: Grid search for optimal parameters
- [ ] **Baseline Comparison**: Benchmark against TF-IDF + Logistic Regression, LSTM, and BERT
- [ ] **Multi-label Classification**: Support products with multiple categories
- [ ] **REST API Deployment**: Flask/FastAPI service for production use
- [ ] **Docker Containerization**: Easy deployment and scaling
- [ ] **Real-time Data Pipeline**: Integration with streaming data sources
- [ ] **A/B Testing Framework**: Model performance monitoring in production
- [ ] **Explainability**: Add LIME/SHAP for prediction interpretability

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Vinay Varshigan**

- Email: sjvinay357@gmail.com
- LinkedIn: [linkedin.com/in/vinaysj2003](https://linkedin.com/in/vinaysj2003)
- GitHub: [@macrosensor2022](https://github.com/macrosensor2022)
- Portfolio: [Coming Soon]

---

## 🙏 Acknowledgments

- **Facebook Research** for developing FastText
- E-commerce dataset providers
- Open-source NLP community

---

## 📚 References

- [FastText Official Documentation](https://fasttext.cc/)
- [FastText Paper (Bojanowski et al., 2017)](https://arxiv.org/abs/1607.04606)
- [Text Classification with FastText](https://fasttext.cc/docs/en/supervised-tutorial.html)

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

*Last Updated: February 2026*
