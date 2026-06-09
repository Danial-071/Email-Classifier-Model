# Email Classification Using Logistic Regression

A binary text classification system that filters emails based on personal interest relevance. The model learns which emails match your interests purely from labeled training data — no hardcoded rules or category definitions required.

---

## Problem Statement

Given a dataset of email texts labeled as **relevant (1)** or **not relevant (0)**, train a logistic regression model to automatically classify new incoming emails.

Relevant emails may span multiple interest domains (e.g. fitness, AI/ML, entrepreneurship). Non-relevant emails may include concerts, dating clubs, societies, etc. The model learns the distinction implicitly from the data.

---

## How It Works

1. **Dataset** — Email texts paired with binary labels (1 = relevant, 0 = not relevant)
2. **Preprocessing** — Lowercase conversion, punctuation removal, stop word stripping
3. **Vectorization** — TF-IDF transforms raw text into numerical feature vectors
4. **Training** — Logistic regression learns word weights that separate the two classes
5. **Inference** — New emails are vectorized and passed through the model; a probability ≥ 0.5 → label 1, otherwise → label 0

---

## Project Structure

```
email-classifier/
├── data/
│   └── emails.csv          # Labeled email dataset (text, label)
├── models/
│   └── classifier.pkl      # Saved trained model
├── src/
│   ├── preprocess.py       # Text cleaning utilities
│   ├── train.py            # Model training script
│   └── predict.py          # Inference on new emails
├── notebooks/
│   └── exploration.ipynb   # EDA and experimentation
├── requirements.txt
└── README.md
```

---

## Dataset Format

| email_text | label |
|---|---|
| Seminar on Large Language Models and Transformer Architectures. | 1 |
| Startup funding strategies for early-stage entrepreneurs. | 1 |
| Book your tickets for the live music concert this Friday. | 0 |
| Join our dating club and find your perfect match. | 0 |

---

## Installation

```bash
git clone https://github.com/your-username/email-classifier.git
cd email-classifier
pip install -r requirements.txt
```

---

## Usage

**Train the model**
```bash
python src/train.py --data data/emails.csv
```

**Classify new emails**
```bash
python src/predict.py --input "Register now for our AI and machine learning bootcamp."
```

---

## Requirements

```
scikit-learn
pandas
numpy
joblib
```

---

## Key Design Decisions

**Binary classification, not multi-class** — The model does not predict email genre. It only answers: does this email match my interests? All interest genres (fitness, AI, entrepreneurship) are collapsed into label 1.

**Implicit category learning** — The model is never told which categories are interesting. It infers this from word frequency patterns across labeled examples. Words like `workout`, `LLM`, `funding` acquire positive weights; words like `tickets`, `venue`, `dating` acquire negative weights.

**TF-IDF over Bag of Words** — Common uninformative words are down-weighted automatically, giving more signal to domain-specific terms.

---

## Limitations

- Logistic regression is a linear model. If your interests span very different topics, the decision boundary may not generalize well. Consider trying **Random Forest** or **SVM** as alternatives.
- Model performance depends heavily on label quality and dataset size. Aim for balanced classes (roughly equal 0s and 1s).

---

## License

MIT
