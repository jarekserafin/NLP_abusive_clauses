**A uni project that uses NLP to categorize a clause as abusive or correct agreement statement based on its content.
Data: https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl
**# Wykrywanie klauzul abuzywnych (PL)

Klasyfikator binarny wykrywający klauzule abuzywne w języku polskim na podstawie publicznego zbioru danych.

## Problem
W umowach konsumenckich występują zapisy naruszające prawa konsumenta (tzw. klauzule abuzywne). Celem projektu jest zbudowanie modelu, który odróżnia klauzule abuzywne od nieabuzywnych.

## Dane
- Źródło: Hugging Face – `laugustyniak/abusive-clauses-pl`
- Rozmiar: ok. 9,3k rekordów; gotowe podziały train/val/test
- Język: polski
- Licencja: **CC BY-NC-SA 4.0** (wyłącznie do celów niekomercyjnych)
> Uwaga: ten projekt ma charakter edukacyjny/demowy.

## Modele
1. **Baseline**: TF‑IDF + Logistic Regression  
2. **Lepszy klasyczny**: TF‑IDF + Linear SVM  
3. (Opcjonalnie) **Transformer (PL)**: np. HerBERT – fine‑tuning

## Jak uruchomić
```bash
# 1) środowisko
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) trening (wybór modelu: lr | svm)
python src/train.py --model svm --seed 42

# 3) ewaluacja
python src/eval.py --checkpoint artifacts/model.joblib

# 4) demo (opcjonalnie)
python app.py  # uruchamia Streamlit/Gradio
