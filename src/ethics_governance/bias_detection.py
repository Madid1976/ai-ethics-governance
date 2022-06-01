import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

def create_dummy_data(num_samples=1000):
    """
    Creates a dummy dataset for bias detection demonstration.
    """
    np.random.seed(42)
    data = {
        'feature1': np.random.rand(num_samples),
        'feature2': np.random.randint(0, 100, num_samples),
        'sensitive_attr': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]), # 0: majority, 1: minority
        'label': np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    # Introduce some bias: minority group has lower chance of positive label
    df.loc[df['sensitive_attr'] == 1, 'label'] = np.random.choice([0, 1], df[df['sensitive_attr'] == 1].shape[0], p=[0.8, 0.2])
    return df

def train_biased_model(df):
    """
    Trains a simple logistic regression model on the potentially biased data.
    """
    X = df[['feature1', 'feature2', 'sensitive_attr']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_pred):.4f}")
    return model, X_test, y_test

def detect_bias(model, X_test, y_test, sensitive_attribute='sensitive_attr', privileged_groups=[{'sensitive_attr': 0}], unprivileged_groups=[{'sensitive_attr': 1}]):
    """
    Detects bias in the model's predictions using AIF360 metrics.
    """
    # Create AIF360 dataset
    aif_data = BinaryLabelDataset(df=X_test.copy(), label_names=['label'], 
                                  protected_attribute_names=[sensitive_attribute],
                                  privileged_classes=[[0]], # Assuming 0 is privileged
                                  favorable_label=1, unfavorable_label=0)
    
    # Add true labels to the dataset
    aif_data.labels = y_test.values.reshape(-1, 1)

    # Add model predictions to the dataset
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    aif_data.scores = y_pred_proba.reshape(-1, 1)
    aif_data.predictions = (y_pred_proba > 0.5).astype(float).reshape(-1, 1)

    metric = ClassificationMetric(aif_data, 
                                  privileged_groups=privileged_groups, 
                                  unprivileged_groups=unprivileged_groups)

    print("\n--- Bias Detection Metrics ---")
    print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")
    print(f"Disparate Impact: {metric.disparate_impact():.4f}")
    print(f"Equal Opportunity Difference: {metric.equal_opportunity_difference():.4f}")
    print(f"Average Odds Difference: {metric.average_odds_difference():.4f}")
    print("------------------------------")

    if abs(metric.statistical_parity_difference()) > 0.1 or metric.disparate_impact() < 0.8 or metric.disparate_impact() > 1.25:
        print("Potential bias detected!")
    else:
        print("No significant bias detected based on these metrics.")

if __name__ == '__main__':
    import numpy as np
    import os

    # Ensure aif360 is installed
    try:
        from aif360.datasets import BinaryLabelDataset
    except ImportError:
        print("AIF360 not installed. Please install with `pip install aif360`")
        print("Skipping bias detection example.")
        exit()

    df = create_dummy_data()
    model, X_test, y_test = train_biased_model(df)
    detect_bias(model, X_test, y_test)
