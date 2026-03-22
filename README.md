# Smart Loan Risk Analyzer

A production-ready Machine Learning application that predicts loan approval risk using applicant financial and demographic data.

## 🚀 Overview
This project provides a robust solution for financial institutions to assess the risk of loan applications. It uses a tuned **Random Forest Classifier** to analyze multiple factors and provide a probability-based risk assessment.

## ⚙️ How It Works
1.  **Data Processing**: Incoming data is cleaned and preprocessed using a scikit-learn pipeline (treating missing values, encoding categorical data, and scaling numerical features).
2.  **Model Inference**: The pre-trained pipeline processes applicant details and outputs a binary classification (Low Risk / High Risk) along with a confidence score.
3.  **Dynamic Dashboard**: An interactive Streamlit interface displays the prediction, feature importance, and model evaluation metrics.

## 📊 Parameters & Features
The model considers the following parameters:
- **Demographics**: Gender, Marital Status, Dependents, Education.
- **Financials**: Applicant Income, Co-applicant Income, Loan Amount, Loan Term.
- **History**: Credit History (Strongest predictor), Employment Type (Self-Employed).
- **Environment**: Property Area (Urban, Semi-urban, Rural).

## 📈 Accuracy & Calculation
- **Model Accuracy**: **86.2%** (Improved from 79% baseline).
- **ROC-AUC Score**: **0.80**.
- **Calculation**: Accuracy is calculated on a held-out test set (20% of original data) that the model never saw during training. The metrics are generated using a 5-fold cross-validation strategy during the hyperparameter tuning phase to ensure robustness.

## 🛠 Tech Stack
- **Languages**: Python
- **ML Frameworks**: Scikit-Learn (Pipelines, RandomForest)
- **Data Handling**: Pandas, NumPy
- **Visuals**: Plotly, Seaborn, Matplotlib
- **Web App**: Streamlit

## 📦 Author
- **Rahul Kumar**

---
## 💻 Setup & Execution
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train the Model** (optional):
    ```bash
    python src/train_model.py
    ```
3.  **Run the App**:
    ```bash
    python -m streamlit run app/streamlit_app.py
    ```
    *Note: If you are using Python 3.14 and encounter an `asyncio` error, please run:*
    ```bash
    python run_app.py
    ```