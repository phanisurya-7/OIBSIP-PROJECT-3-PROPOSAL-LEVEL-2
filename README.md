# OIBSIP-PROJECT-3-PROPOSAL-LEVEL-2


# Fraud Detection Project

## Overview
This project implements a **fraud detection system** to classify credit card transactions as fraudulent or legitimate using machine learning models. The goal is to accurately predict fraudulent transactions based on anonymized features, enabling faster detection of fraud in financial systems.

## Dataset
The dataset used for this project is the **Credit Card Fraud Detection Dataset** available on Kaggle. This dataset contains credit card transactions from September 2013, where each transaction is classified as fraudulent (`Class = 1`) or legitimate (`Class = 0`).

You can download the dataset here:
- **Dataset**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Since the dataset is large, it is not included directly in this repository. Please download the dataset from Kaggle and place the `creditcard.csv` file in the project directory before running the code.

## Key Concepts
- **Anomaly Detection**: Identifying transactions that deviate significantly from normal behavior (fraudulent transactions).
- **Machine Learning Models**: We use Logistic Regression, Decision Trees, and other classifiers to predict whether a transaction is fraudulent or legitimate.
- **Feature Engineering**: The dataset is preprocessed by scaling the features and handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
- **Real-time Monitoring**: While this project focuses on offline analysis, real-time monitoring can be explored in future work to detect fraud as it happens.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
   ```

2. Download the dataset from Kaggle and place the `creditcard.csv` file in the project directory.

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Python script:
   ```bash
   python fraud_detection.py
   ```

## Evaluation Metrics
- **Precision, Recall, F1-Score**: These metrics are used to evaluate the model's ability to detect fraudulent transactions and minimize false positives/negatives.
- **ROC AUC Score**: This score measures the model's ability to distinguish between fraudulent and legitimate transactions.

## Tools & Libraries Used
- **Python 3.x**
- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical operations.
- **Scikit-Learn**: For machine learning models and preprocessing.
- **Imbalanced-learn**: For handling class imbalance.
- **Matplotlib & Seaborn**: For visualizations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Kaggle for providing the Credit Card Fraud Detection Dataset.
- Scikit-learn for machine learning tools.
- Imbalanced-learn for handling class imbalance in the dataset.
```
