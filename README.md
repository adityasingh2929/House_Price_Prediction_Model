# House Price Prediction Model

This project implements a machine learning based **house price prediction system** using the California Housing dataset.  
It demonstrates an **end to end regression workflow**, covering data preprocessing, feature engineering, model training, evaluation, persistence, and inference.

---

## 1. Motivation behind the project

House prices depend on multiple factors such as location, population, income levels, and housing characteristics.  
Predicting house prices is a classic **regression problem** and is widely used in real world applications like real estate valuation and market analysis.

This project was built to:
- Practice real world data preprocessing
- Learn how to use preprocessing pipelines correctly
- Train and evaluate a Random Forest regression model
- Understand model evaluation using RMSE
- Build a clean, reproducible, GitHub ready ML project

---

## 2. Concepts and techniques used

### Data preprocessing pipeline
A complete preprocessing pipeline was built to ensure consistent transformations during both training and inference.

The pipeline includes:
- **Handling missing values**
  - `SimpleImputer` for numerical features
- **Feature scaling**
  - Standardization using `StandardScaler`
- **Categorical feature handling**
  - One hot encoding for categorical features
- **Data cleaning and feature engineering**
  - Derived meaningful ratios from raw attributes

### Train test splitting
- **Stratified Shuffle Split**
  - Preserves important feature distributions
  - Helps improve generalization performance

### Model used
- **RandomForestRegressor**
  - Handles non linear relationships effectively
  - Reduces overfitting compared to single decision trees
  - Performs well on structured tabular data

---

## 3. Model performance and efficiency

The model was evaluated using **Root Mean Squared Error (RMSE)**.


### Interpretation
- On average, predicted house prices differ from actual prices by about **47,000 USD**
- Given that house prices range up to roughly 500,000 USD, this performance is reasonable for a baseline Random Forest model
- The model captures meaningful patterns but can be further improved with tuning and better feature engineering

---

## 4. Model and pipeline generation logic

This project is designed to automatically handle model creation and reuse.

- On the **first run**:
  - The model is trained
  - The full preprocessing pipeline is built
  - Both are saved locally as:
    - `model.pkl`
    - `pipeline.pkl`

- On **subsequent runs**:
  - If the saved model and pipeline already exist, the code:
    - Loads them using `joblib`
    - Skips retraining
    - Directly performs inference on input data

This avoids unnecessary retraining and ensures faster execution after the initial run.

> Note: The trained model and pipeline files are not committed to GitHub due to file size limitations.  
> They are generated locally when the project is executed.

---

## 5. Generated input and output files

During execution, the project creates and uses the following CSV files:

- `input.csv`
- `input_validation.csv`
- `output.csv`

These files are **generated or modified by the code at runtime** and are therefore not committed to the repository.

They are ignored using `.gitignore` to:
- Keep the repository clean
- Avoid committing generated artifacts
- Ensure reproducibility through code instead of static files

Each run can regenerate these files as needed.

---

## 6. Project structure

House_Price_Prediction_Model/
│
├── housing.csv
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── venv/ (ignored)
├── model.pkl (generated locally, ignored)
├── pipeline.pkl (generated locally, ignored)
├── input.csv (generated, ignored)
├── input_validation.csv (generated, ignored)
└── output.csv (generated, ignored)


---

## 7. How to run this project locally


```bash
git clone <your-github-repo-url>
cd House_Price_Prediction_Model
python -m venv venv
.\venv\Scripts\activate   [windows]     (keep in mind that you run this everytime you re-open the project and also that you run this in cmd, not in powershell.)
   [if Mac or Linux]:  source venv/bin/activate
pip install -r requirements.txt
python main.py

```

On first run:

The model and pipeline will be trained and saved locally

On later runs:

Saved artifacts will be loaded

Inference will be performed directly




## 8. Future improvements

Hyperparameter tuning using GridSearchCV

Cross validation for more stable RMSE

Trying Gradient Boosting or XGBoost

Improved feature engineering

Model evaluation on unseen datasets


## 9. Learning outcomes

This project helped reinforce:

End to end machine learning workflow

Proper use of preprocessing pipelines

Model evaluation using RMSE

Model persistence using joblib

Reproducible ML projects with Git and GitHub

Writing clear and professional documentation


