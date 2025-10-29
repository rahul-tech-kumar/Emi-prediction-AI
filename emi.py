import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")  # To suppress convergence/future warnings

# ------------------ Load Dataset ------------------
df = pd.read_csv("emi_prediction_dataset.csv", low_memory=False)

# ------------------ Missing Values ------------------
print("Missing values per column:\n", df.isnull().sum())
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Percentage of missing data: {df.isnull().sum().sum() / (df.shape[0]*df.shape[1])*100:.2f}%")

# ------------------ Fill Missing Values ------------------
df['education'] = df['education'].fillna(df['education'].mode()[0])

# Convert bank_balance to numeric
df['bank_balance'] = df['bank_balance'].str.replace(',', '').str.replace(r'\.0$', '', regex=True)
df['bank_balance'] = pd.to_numeric(df['bank_balance'], errors='coerce')

# Imputer for numeric columns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

cols = ["monthly_rent","credit_score","bank_balance","emergency_fund"]
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
df[cols] = imputer.fit_transform(df[cols])

# Age
df['age'] = pd.to_numeric(df['age'].astype(str).str.replace(r'\.0$', '', regex=True), errors='coerce').astype('Int64')

# Monthly salary
df["monthly_salary"] = pd.to_numeric(df["monthly_salary"].astype(str).str.replace(r'\.0$', '', regex=True), errors="coerce")
df["monthly_salary"] = df["monthly_salary"].fillna(df["monthly_salary"].median())

# Existing loans
loan_map = {'Yes': 1, 'No': 0}
df["existing_loans"] = df["existing_loans"].replace(loan_map).astype("Int64")


#Gender in correct format
df['gender'] = df['gender'].str.strip().str.lower()   # normalize case & remove spaces
gender_map = {
    'male': 1, 'm': 1,
    'female': 0, 'f': 0
}

df['gender'] = df['gender'].map(gender_map)

# ------------------ Feature Engineering ------------------
df['total_expense'] = df['groceries_utilities'] + df['school_fees'] + df['travel_expenses'] + df['other_monthly_expenses']
df['expense_income_ratio'] = df['total_expense'] / df['monthly_salary']
df['saving_ratio'] = (df['monthly_salary'] - df['total_expense']) / df['monthly_salary']
df['emi_ratio'] = df['current_emi_amount'] / df['monthly_salary']
df['financial_stability_index'] = df['bank_balance'] + df['emergency_fund']
df['dependents_ratio'] = df['dependents'] / df['family_size']
df['affordability_ratio'] = df['bank_balance'] / df['monthly_salary']


# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['emi_eligibility'] = le.fit_transform(df['emi_eligibility'])
print("Unique labels:", df['emi_eligibility'].unique())



# Categorical encoding
#  One-Hot Encode Categorical Columns
df = pd.get_dummies(
    df,
    columns=["gender", "marital_status", "education", "employment_type", "company_type", "house_type", "emi_scenario"],
    drop_first=False  # keep all dummy columns to avoid info loss
)

#  Scale Numeric Columns
from sklearn.preprocessing import StandardScaler

num_cols = [
    'age', 'monthly_salary', 'years_of_employment', 'credit_score',
    'bank_balance', 'emergency_fund', 'requested_amount', 'requested_tenure',
    'emi_ratio', 'affordability_ratio', 'dependents_ratio',
    'saving_ratio', 'expense_income_ratio'
]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#  Automatically detect encoded columns
# (so you don’t have to hardcode names)
encoded_cols = [col for col in df.columns if any(prefix in col for prefix in [
    'gender_', 'marital_status_', 'education_', 'employment_type_',
    'company_type_', 'house_type_', 'emi_scenario_'
])]

#  Combine numeric + encoded features
feature_cols = num_cols + encoded_cols



# ------------------ Train-Test Split for classification ------------------

from sklearn.model_selection import train_test_split
X = df[feature_cols]
y = df['emi_eligibility']

# classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y, test_size=0.2, random_state=42)



# ------------------Train-Test Split for regression------------------------------------------
X = df[feature_cols]
y_reg = df['max_monthly_emi']


#regression
X_train_reg,X_test_reg, y_train_reg, y_test_reg = train_test_split(X,y_reg,test_size=0.2,random_state=42)



# ------------------ MLflow Setup --------------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5005")
mlflow.set_experiment("EMIPredict_Classification")

# ------------------ Models ---------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score,confusion_matrix,r2_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

models = [
    # ---------------------classification-------------------------
    ("Logistic Regression", {"solver": "liblinear", "random_state": 42}, LogisticRegression()),
    ("Random Forest", {"n_estimators": 200, "max_depth": 5, "random_state": 42, "n_jobs": -1}, RandomForestClassifier()),
    ("XGBoost Classifier", {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}, XGBClassifier()),
    ("Decision Tree Classifier", {"max_depth": None, "random_state": 42}, DecisionTreeClassifier())
]

# Regression Models
# ------------------------------------
regression_models = [
    ("Linear Regression",
     {"fit_intercept": True},
     LinearRegression()),

    ("Random Forest Regressor",
     {"n_estimators": 150, "max_depth": 8, "random_state": 42},
     RandomForestRegressor()),

    ("XGBoost Regressor",
     {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6, "random_state": 42},
     XGBRegressor())
]



# For classification
results = []

for model_name, params, model in models:
    with mlflow.start_run(run_name=model_name):
        print(f"\n🚀 Training {model_name}...")
        # Create subfolder for each model artifacts
        model_artifact_dir = f"artifacts/{model_name.replace(' ', '_')}"
        os.makedirs(model_artifact_dir, exist_ok=True)
        
        # Set hyperparameters and fit
        model.set_params(**params)
        model.fit(X_train_cls, y_train_cls)
        y_pred = model.predict(X_test_cls)
        
        # Metrics
        acc = accuracy_score(y_test_cls, y_pred)
        f1 = f1_score(y_test_cls, y_pred, average="weighted")
        mse=mean_squared_error(y_test_cls,y_pred)
        mae=mean_absolute_error(y_test_cls,y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print(f"✅ F1 Score: {f1:.4f}")
        print(f"✅ MSE: {mse}")
        print(f"✅ MAE: {mae}")
        
        print("📊 Classification Report:")
        print(classification_report(y_test_cls, y_pred))
        
        # MLflow logging
        #Log comprehensive model parameters, hyperparameters, and performance metrics for all models
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.sklearn.log_model(model, "model")
        
        
        #  Log Confusion Matrix
        
        cm = confusion_matrix(y_test_cls, y_pred)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        
        cm_path = os.path.join(model_artifact_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        
        # Log Model
        # -----------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="classification_model",
            registered_model_name=model_name.replace(" ", "_") + "_Classification"
        )

        
        
        # Creating a Dataframe for comparision
        results.append({"Model": model_name,
                        "Accuracy": acc,
                        "F1score": f1,
                        "MSE": mse,
                        "MAE": mae,
                        "run_id": mlflow.active_run().info.run_id})
        
        
        
#  Best Model Based on F1-Score
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
best_model_row = results_df.loc[results_df['F1score'].idxmax()]
best_model_name = best_model_row["Model"]
best_run_id = best_model_row["run_id"]
best_f1 = best_model_row["F1score"]

print("\n🏆 Best Model Selected:")
print(f"Model: {best_model_name}")
print(f"Run ID: {best_run_id}")
print(f"F1-Score: {best_f1:.4f}")

# -----------------------------------------------------
#  Register Best Model for Production
# -----------------------------------------------------
# Load the best model from its MLflow run and register it for production use
from mlflow.tracking import MlflowClient    # which is the main API for interacting programmatically with MLflow’s tracking server and model registry.
client = MlflowClient()

model_uri = f"runs:/{best_run_id}/model"  #URI- Unique Path
registered_name = best_model_name.replace(" ", "_") + "_Classification_Production"

# Register model
registered_model = mlflow.register_model(model_uri=model_uri, name=registered_name)

print(f"\n🚀 Best model registered to MLflow Model Registry as: {registered_name}")
print("You can now view or deploy it from the MLflow UI.")

print("\n🎯 Complete pipeline executed: models trained, compared, and best model registered!")




# -------------------------------------------------Regression----------------------------------------------------------------------------

#  MLflow Experiment
# ------------------------------------
mlflow.set_experiment("EMI_Regression_Model")

results_reg = []

for name, params, model in regression_models:
    with mlflow.start_run(run_name=name):
        model.set_params(**params)
        model.fit(X_train_reg, y_train_reg)
        y_pred_reg = model.predict(X_test_reg)

        # -----------------------------
        # Compute Regression Metrics
        # -----------------------------
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        rmse = mse ** 0.5
        r2 = r2_score(y_test_reg, y_pred_reg)

        # -----------------------------
        # Log Parameters and Metrics
        # -----------------------------
        
        mlflow.log_params(params)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2_score", r2)
        mlflow.sklearn.log_model(model, "model")

        # -----------------------------
        # Plot Predicted vs Actual
        # -----------------------------
        plt.figure(figsize=(5,5))
        sns.scatterplot(x=y_test_reg, y=y_pred_reg, alpha=0.6)
        plt.xlabel("Actual EMI")
        plt.ylabel("Predicted EMI")
        plt.title(f"Predicted vs Actual - {name}")
        plt.plot([y_test_reg.min(), y_test_reg.max()],
                 [y_test_reg.min(), y_test_reg.max()],
                 'r--')
        plt.tight_layout()

        os.makedirs("artifacts/regression_plots", exist_ok=True)
        plot_path = f"artifacts/regression_plots/{name.replace(' ', '_')}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # -----------------------------
        # Log Model
        # -----------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="regression_model",
            registered_model_name=name.replace(" ", "_") + "_Regression"
        )

        # Save result for comparison
        results_reg.append({
            "name": name,
            "R2": r2,
            "RMSE": rmse,
            "run_id": mlflow.active_run().info.run_id
        })

        print(f"✅ {name} logged | R2: {r2:.4f}, RMSE: {rmse:.4f}")

# ------------------------------------
# 4️⃣ Select Best Regression Model
# ------------------------------------
results_reg_df = pd.DataFrame(results_reg)
best_reg = results_reg_df.loc[results_reg_df["R2"].idxmax()]
best_reg_name = best_reg["name"]
best_reg_run_id = best_reg["run_id"]
print("\n🏆 Best Regression Model:")
print(best_reg)


model_uri = f"runs:/{best_reg_run_id}/model"  #URI- Unique Path
registered_name = best_reg_name.replace(" ", "_") + "Regression_Production"

# Register model
registered_model = mlflow.register_model(model_uri=model_uri, name=registered_name)

print(f"\n🚀 Best model registered to MLflow Model Registry as: {registered_name}")
print("You can now view or deploy it from the MLflow UI.")

print("\n🎯 Complete pipeline executed: models trained, compared, and best model registered!")


# ---------------------------------------------------------------------------------

#loading using joblib

import joblib

# Save best models locally for Streamlit app
os.makedirs("models", exist_ok=True)

# Best classification model (reload from MLflow)
best_cls_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
joblib.dump(best_cls_model, "models/best_emi_classifier.pkl")

# Best regression model
best_reg_model = mlflow.sklearn.load_model(f"runs:/{best_reg_run_id}/model")
joblib.dump(best_reg_model, "models/best_emi_regressor.pkl")

print("\n✅ Best models saved locally for Streamlit app in 'models/' folder.")


