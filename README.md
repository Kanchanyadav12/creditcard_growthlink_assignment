# creditcard_growthlink_assignment

## Objective
The objective of this project is to build an efficient machine learning model using **Random Forest Classifier** to detect fraudulent credit card transactions. The dataset is highly imbalanced, containing a very small percentage of fraud cases, which makes it a real-world challenge. The goal is to minimize false positives while maintaining high fraud detection accuracy.



## Dataset Overview
- Features `V1` to `V28` are results of a PCA transformation to protect sensitive information.
- Additional features include:
  - **Time**: Time in seconds since the first transaction.
  - **Amount**: Transaction amount.
  - **Class**: Target variable (1 = fraud, 0 = non-fraud).


## Steps to Run

1. **Clone the repository:**
```bash
git clone https://github.com/Kanchanyadav12/creditcard_growthlink_assignment
cd creditcard_growthlink_assignment
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Jupyter Notebook:**
```bash
jupyter notebook credit_card_.ipynb
```

---

## Workflow Summary

### 1. Exploratory Data Analysis (EDA)
- Checked class distribution (0 vs 1) to confirm severe imbalance.
- Visualized transaction **Amount** and **Time** distributions.
- Engineered new feature: `Hour` from `Time`.

### 2. Feature Engineering
- Scaled the `Amount` feature using **StandardScaler**.
- Extracted transaction `Hour` from `Time`.
- Dropped `Time` and original `Amount` after transformation.
- Applied **outlier capping** on `scaled_amount` using IQR.

### 3. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to oversample the minority class and balance the dataset.

### 4. Model Training (Random Forest Only)
- Used `RandomForestClassifier(n_estimators=1000)`.
- Trained the model on SMOTE-resampled data.
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - ROC-AUC Curve

### 5. ROC Curve
- Plotted ROC curve to visualize tradeoff between TPR and FPR.
- Random Forest achieved a high AUC, indicating strong predictive performance.

---

## Results
| Metric      | Value (Example) |
|-------------|------------------|
| Accuracy    | 0.9992           |
| Precision   | 0.88             |
| Recall      | 0.91             |
| F1 Score    | 0.89             |
| AUC Score   | 0.98             |

> Note: Actual results may vary slightly depending on train-test split and random seed.

---

## Highlights & Innovations
- Used **SMOTE** to deal with class imbalance.
- Introduced time-based feature `Hour` for temporal fraud analysis.
- Used outlier capping on monetary feature to prevent skew.
- Focused on Random Forest for its interpretability and robustness.

---

## Contact
For queries or contributions, feel free to reach out:
- Email: kanchanyadav4065@gmail.com
- GitHub: [your-username](https://github.com/Kanchanyadav12)
