# 🩺 Diabetes Risk Predictor

A simple machine learning web app that estimates a person's risk of diabetes based on basic health information (age, BMI, blood glucose, HbA1c, etc.). Built with **scikit-learn** and **Streamlit**.

⚠️ **Disclaimer:** This is a learning/demo project trained on a public dataset. It is **not a medical diagnostic tool** and should not be used for real health decisions.

---

## 🔍 What it does

1. You enter basic health details in a form (gender, age, BMI, glucose level, etc.)
2. A trained Random Forest model predicts the probability of diabetes
3. The app shows the risk percentage and flags the person as **lower risk** or **higher risk** (using an adjustable decision threshold, default 0.50)

---

## 📁 Project files

| File                   | What it does                                                       |
| ---------------------- | ------------------------------------------------------------------ |
| `train.py`             | Cleans the dataset, trains the model, and saves it as `model.pkl`  |
| `app.py`               | The Streamlit web app that loads `model.pkl` and makes predictions |
| `model.pkl`            | The already-trained model (ready to use — no need to retrain)      |
| `requirements.txt`     | List of Python packages needed to run the app                      |
| `diabetes_dataset.csv` | The training data (only needed if you want to retrain)             |
| `Final_ML_Notebook.html` | Exploratory notebook comparing Random Forest, XGBoost, Logistic Regression, KNN, Decision Tree, and Naive Bayes on the full feature set (unbalanced) — the deployed model below is a separate, tuned version built specifically for the app |

---

## 🚀 Run it locally

**Step 1 — Install the requirements**

```
pip install -r requirements.txt
```

**Step 2 — Run the app**

```
streamlit run app.py
```

**Step 3 — Open your browser**
It will automatically open at `http://localhost:8501`

---

## 🌍 Deploy it online (free, ~2 minutes)

1. Push this repo to GitHub (make sure `app.py`, `model.pkl`, and `requirements.txt` are included)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set main file to `app.py`
4. Click **Deploy**

You'll get a public link like `yourapp.streamlit.app` that anyone can open and use.

---

## 🔄 Retrain the model (optional)

If you want to retrain on new or updated data:

```
python train.py
```

This regenerates `model.pkl` using `diabetes_dataset.csv`.

---

## 🧠 About the model

- **Algorithm:** Random Forest Classifier (`n_estimators=300`, `max_depth=12`, `random_state=42`)
- **Features used:** gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level
  *(location, year, and race columns are intentionally excluded from the deployed model — dropped to keep the input form simple and to avoid using race as a predictive feature)*
- **Preprocessing:** StandardScaler fit on the training split only (avoids leaking test-set information, unlike the exploratory notebook)
- **Class imbalance handling:** `class_weight="balanced"` (diabetes cases are only ~8.5% of the data)
- **Verified performance on held-out test set (20,000 records):**

  | Metric | Score |
  |---|---|
  | Accuracy | 91.6% |
  | Precision (diabetic class) | 50.4% |
  | Recall (diabetic class) | 89.1% |
  | F1 score (diabetic class) | 0.64 |

  This is a deliberate trade-off: `class_weight="balanced"` sacrifices precision (more false positives) in exchange for much higher recall, so the model misses far fewer true diabetic cases — appropriate for a health-screening context where a missed case is costlier than a false alarm.

---

## 🛠️ Built with

- [Python](https://python.org)
- [scikit-learn](https://scikit-learn.org)
- [Streamlit](https://streamlit.io)
- [pandas](https://pandas.pydata.org)
