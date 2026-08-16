# 🎓 AI Dependency Classifier

Predicting student AI dependency level (**Low / Medium / High**) from behavioral, academic, and demographic survey data — a machine learning classification project built end-to-end in Jupyter.

![Project cover](dataset-cover.png)

---

## 📌 Overview

As AI tools become part of daily student life, understanding *how much* a student relies on them can help institutions offer better guidance and support — before over-reliance affects independent thinking and skill development.

This project uses survey data from **15,000 students** to classify AI dependency into three levels, using a fully leakage-safe, cross-validated machine learning pipeline.

---

## 📊 Results

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Baseline (majority class) | 0.4297 | 0.2583 |
| **Final: Tuned Logistic Regression** | **0.6613** | **0.6620** |

- **+0.4037 F1** improvement over baseline — the model learned real, generalizable patterns rather than exploiting class imbalance.
- **CV-to-test gap: only 1.6%** (0.6729 → 0.6620) — the model generalizes well to unseen students.

**Per-class performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| High | 0.73 | 0.60 | 0.66 |
| Low | 0.71 | 0.70 | 0.70 |
| Medium | 0.60 | 0.66 | 0.63 |

---

## 🔑 Key Modeling Decisions

1. **Leakage prevention** — dropped `ai_dependency_score` (used to create the target label) and `ai_replaces_own_thinking_score` (measures a near-identical construct) before training.
2. **Stratified splitting & cross-validation** — used throughout, given class imbalance (Low 36% / Medium 43% / High 21%).
3. **Fair model comparison** — 6 models (Dummy, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM) evaluated under identical pipeline + cross-validation conditions.
4. **Hyperparameter tuning** — `BayesSearchCV` over `C` and `l1_ratio` (elastic-net mix), yielding only a marginal gain — evidence the model was already near its performance ceiling for this dataset.

---

## ⚠️ Key Limitation

The confusion matrix shows the model's errors are concentrated between **adjacent classes** (High↔Medium, Low↔Medium), with almost no confusion between the two extremes — suggesting the model implicitly learned the ordinal structure of dependency, despite being trained as an unordered classifier.

Most notable weakness: **38% of truly High-dependency students (242 of 630) were misclassified as Medium** — meaning a real-world early-intervention use case would miss a meaningful share of at-risk students.

---

## 🚧 Future Work

- Try an **ordinal classification** approach instead of treating classes as unordered
- Engineer features that better distinguish Medium from High dependency
- Explore class-weighting or resampling to improve High-class recall specifically

---

## 🗂️ Project Structure

```
├── project.ipynb              # Full analysis: EDA → cleaning → modeling → evaluation
├── project.html               # Full project as a HTML page
├── ai_dependency_model.pkl    # Exported, trained pipeline (preprocessing + model)
├── Report_AI_dependency.html  # Sweetviz data profiling report
├── README.md
├── LICENSE                    # MIT
└── .gitignore
```

---

## 🛠️ Tech Stack

`pandas` · `numpy` · `scikit-learn` · `scikit-optimize (BayesSearchCV)` · `matplotlib` / `seaborn` · `sweetviz` · `joblib`

---

## 🔄 Reproducing This Project

```bash
- git clone <your-repo-url>
- cd AI_Dependency
- python -m venv venv
- pip install -r requirements.txt   # or install packages listed at the top of project.ipynb
- jupyter lab project.ipynb
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.