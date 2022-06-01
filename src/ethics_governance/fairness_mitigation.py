import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

from ethics_governance.bias_detection import create_dummy_data, detect_bias

def reweigh_data(df, sensitive_attribute=\'sensitive_attr\', privileged_groups=[{\'sensitive_attr\': 0}], unprivileged_groups=[{\'sensitive_attr\': 1}]):
    """
    Applies reweighing as a pre-processing bias mitigation technique.
    """
    print("\nApplying Reweighing mitigation...")
    aif_data = BinaryLabelDataset(df=df, label_names=[\'label\'], 
                                  protected_attribute_names=[sensitive_attribute],
                                  privileged_classes=[[0]], 
                                  favorable_label=1, unfavorable_label=0)

    RW = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    aif_data_reweighed = RW.fit_transform(aif_data)
    
    # Convert back to pandas DataFrame for model training
    df_reweighed = aif_data_reweighed.convert_to_dataframe()
    return df_reweighed

def train_mitigated_model(df_mitigated):
    """
    Trains a logistic regression model on the reweighed data.
    """
    X = df_mitigated[[\'feature1\', \'feature2\', \'sensitive_attr\']]
    y = df_mitigated[\'label\']
    sample_weights = df_mitigated[\'instance_weights\'] # Weights from Reweighing

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y, sample_weights, test_size=0.3, random_state=42)

    model = LogisticRegression(solver=\'liblinear\', random_state=42)
    model.fit(X_train, y_train, sample_weight=sw_train)
    y_pred = model.predict(X_test)
    print(f"Mitigated model accuracy: {accuracy_score(y_test, y_pred):.4f}")
    return model, X_test, y_test

def apply_prejudice_remover(df, sensitive_attribute=\'sensitive_attr\'):
    """
    Applies Prejudice Remover as an in-processing bias mitigation technique.
    """
    print("\nApplying Prejudice Remover mitigation...")
    aif_data = BinaryLabelDataset(df=df, label_names=[\'label\'], 
                                  protected_attribute_names=[sensitive_attribute],
                                  privileged_classes=[[0]], 
                                  favorable_label=1, unfavorable_label=0)
    
    # Split data for PrejudiceRemover
    (train_data, test_data) = aif_data.split([0.7], shuffle=True)

    # Train PrejudiceRemover model
    PR = PrejudiceRemover(eta=0.8, sensitive_attribute=sensitive_attribute)
    PR.fit(train_data)
    
    # Get predictions on test data
    test_data_pred = PR.predict(test_data)
    
    # Convert back to pandas for evaluation
    X_test = test_data.features
    y_test = test_data.labels.ravel()
    y_pred = test_data_pred.labels.ravel()

    # Create a dummy model object for detect_bias compatibility
    class DummyModel:
        def predict(self, X): return y_pred
        def predict_proba(self, X): return np.column_stack([1-y_pred, y_pred])
    
    print(f"Prejudice Remover model accuracy: {accuracy_score(y_test, y_pred):.4f}")
    return DummyModel(), pd.DataFrame(X_test, columns=test_data.feature_names), pd.Series(y_test)

if __name__ == \'__main__\':
    import numpy as np
    import os

    # Ensure aif360 is installed
    try:
        from aif360.datasets import BinaryLabelDataset
    except ImportError:
        print("AIF360 not installed. Please install with `pip install aif360`")
        print("Skipping fairness mitigation example.")
        exit()

    df = create_dummy_data()
    print("\n--- Original Model Bias ---")
    original_model, X_test_orig, y_test_orig = train_mitigated_model(df) # Using train_mitigated_model for consistency in output format
    detect_bias(original_model, X_test_orig, y_test_orig)

    # Reweighing example
    df_reweighed = reweigh_data(df)
    reweighed_model, X_test_rw, y_test_rw = train_mitigated_model(df_reweighed)
    print("\n--- Reweighed Model Bias ---")
    detect_bias(reweighed_model, X_test_rw, y_test_rw)

    # Prejudice Remover example
    pr_model, X_test_pr, y_test_pr = apply_prejudice_remover(df)
    print("\n--- Prejudice Remover Model Bias ---")
    detect_bias(pr_model, X_test_pr, y_test_pr)
