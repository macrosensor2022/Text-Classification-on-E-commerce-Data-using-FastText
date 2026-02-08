# Text Classification on E-commerce Data using FastText

A text classification system using Facebook's FastText library for categorizing e-commerce product descriptions.

## About

This project uses FastText to classify product titles and descriptions into different categories. FastText is fast to train and handles typos/misspellings well, which is common in e-commerce data. The model uses subword embeddings which helps it understand words it hasn't seen before during training.

## Why FastText?

- Trains much faster compared to deep learning models like LSTM or BERT
- Works well with typos and rare words using character n-grams
- Low memory usage and fast predictions
- Good accuracy for text classification tasks

## Features

- Multi-class classification for product categories
- Preprocessing pipeline for cleaning text data
- Subword embeddings for handling unknown words
- Jupyter notebook for experimentation
- Evaluation metrics and confusion matrix

## Technical Details

**Pipeline:**
1. Load and clean e-commerce text data
2. Convert to FastText format (`__label__<category> <text>`)
3. Train model with word and character n-grams
4. Evaluate on test set
5. Make predictions on new data

**Model settings:**
- Embedding dimension: 100
- Learning rate: 0.1 (tunable)
- Word n-grams: up to 2
- Loss: Softmax

## Dataset

The dataset contains e-commerce product descriptions in CSV format with text and category labels. Categories include things like Electronics, Fashion, Home & Kitchen, etc.

## Installation

**Requirements:**
- Python 3.7+

**Setup:**

```bash
git clone https://github.com/macrosensor2022/Text-Classification-on-E-commerce-Data-using-FastText.git
cd Text-Classification-on-E-commerce-Data-using-FastText
pip install -r requirements.txt
```

Main libraries: `fasttext`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `jupyter`

## Usage

Open the Jupyter notebook to see the full workflow:

```bash
jupyter notebook Text_classification_fasttext_ecommerce_dataset.ipynb
```

Or train a model programmatically:
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

## Results

The model trains fast and achieves good accuracy on the test set. It handles typos well (like "wireles headphons" instead of "wireless headphones") thanks to character n-grams. Check the `output/` folder for visualization of results.

## Project Structure

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

## Future Ideas

- Hyperparameter tuning for better accuracy
- Compare with other models (TF-IDF, LSTM, BERT)
- Multi-label classification support
- Deploy as REST API

## References

- [FastText Documentation](https://fasttext.cc/)
- [FastText Paper](https://arxiv.org/abs/1607.04606)
