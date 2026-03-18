import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic data to simulate a scenario where a model might exhibit bias.
    Scenario: Loan application approval based on age, income, and a protected attribute (gender).
    """
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 60, num_samples),
        'income': np.random.randint(30000, 120000, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'credit_score': np.random.randint(300, 850, num_samples)
    }
    df = pd.DataFrame(data)

    # Introduce synthetic bias: Females with lower income are less likely to get a high credit score
    # This is a simplification to demonstrate bias detection
    df['approved'] = 0 # Default to not approved
    df.loc[(df['income'] > 60000) & (df['credit_score'] > 650), 'approved'] = 1
    df.loc[(df['gender'] == 'Female') & (df['income'] < 50000) & (df['credit_score'] < 600), 'approved'] = 0
    df.loc[(df['gender'] == 'Female') & (df['income'] > 70000) & (df['credit_score'] > 700), 'approved'] = 1

    # Ensure some balance in approved/not approved
    df.loc[df['approved'] == 0, 'approved'] = np.random.choice([0, 1], df[df['approved'] == 0].shape[0], p=[0.8, 0.2])

    return df

def train_biased_model(df):
    """
    Trains a logistic regression model on the synthetic data and evaluates it.
    """
    print("Generating synthetic data with potential bias...")
    
    # Define features and target
    X = df[['age', 'income', 'gender', 'credit_score']]
    y = df['approved']

    # Define preprocessing steps for numerical and categorical features
    numeric_features = ['age', 'income', 'credit_score']
    categorical_features = ['gender']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create a pipeline with preprocessing and logistic regression model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    print("Evaluating model performance...")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    return model_pipeline, X_test, y_test, y_pred

def analyze_bias(X_test, y_test, y_pred, protected_attribute='gender'):
    """
    Analyzes potential bias in model predictions based on a protected attribute.
    """
    print(f"\nAnalyzing bias for protected attribute: {protected_attribute}")
    results = X_test.copy()
    results['true_label'] = y_test
    results['predicted_label'] = y_pred

    for attribute_value in results[protected_attribute].unique():
        subset = results[results[protected_attribute] == attribute_value]
        if not subset.empty:
            accuracy_subset = accuracy_score(subset['true_label'], subset['predicted_label'])
            report_subset = classification_report(subset['true_label'], subset['predicted_label'], output_dict=True)
            print(f"\n--- Metrics for {protected_attribute} = {attribute_value} ---")
            print(f"Accuracy: {accuracy_subset:.4f}")
            print(f"Precision (approved): {report_subset['1']['precision']:.4f}")
            print(f"Recall (approved): {report_subset['1']['recall']:.4f}")
            print(f"F1-Score (approved): {report_subset['1']['f1-score']:.4f}")
        else:
            print(f"No samples for {protected_attribute} = {attribute_value} in test set.")

if __name__ == "__main__":
    print("Starting AI Ethics Bias Detection Pipeline...")

    # 1. Generate synthetic data
    data_df = generate_synthetic_data(num_samples=2000)

    # 2. Train a potentially biased model
    model, X_test, y_test, y_pred = train_biased_model(data_df)

    # 3. Analyze bias in predictions
    analyze_bias(X_test, y_test, y_pred, protected_attribute='gender')

    print("\nAI Ethics Bias Detection Pipeline finished successfully!")
