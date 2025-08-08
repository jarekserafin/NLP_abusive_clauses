# Detection of Abusive Clauses in Polish Contracts

A binary text classification model for detecting abusive clauses in consumer contracts written in Polish, based on a publicly available dataset.

## Problem
Consumer contracts sometimes contain terms that violate consumer rights (abusive clauses). The goal of this project is to build an NLP model that automatically detects whether a given clause is abusive.

## Dataset
- Source: Hugging Face – `laugustyniak/abusive-clauses-pl`
- Size: ~9.3k clauses; predefined train/validation/test splits
- Language: Polish
- License: **CC BY-NC-SA 4.0** (non-commercial use only)
> Disclaimer: This project is for educational/demo purposes only.

## Models
1. **Baseline**: TF-IDF + Logistic Regression  
2. **Improved classic**: TF-IDF + Linear SVM  
3. *(Optional)* **Transformer (PL)**: fine-tuning `HerBERT`

## How to Run
```bash
# 1) Create virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Train (choose model: lr | svm)
python src/train.py --model svm --seed 42

# 3) Evaluate
python src/eval.py --checkpoint artifacts/model.joblib

# 4) Demo (optional)
python app.py  # Streamlit/Gradio UI
