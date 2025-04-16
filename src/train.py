from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = train.drop(columns=["BMI"])
y_train = train["BMI"]
X_test = test.drop(columns=["BMI"])
y_test = test["BMI"]

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5),
    "RandomForest": RandomForestRegressor()
}

mlflow.set_experiment("BMI_Prediction_Experiment")

best_model = None
best_model_name = None
best_mae = float('inf')

# Ensure 'temp_models' folder is clean and created
os.makedirs("temp_models", exist_ok=True)

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        rmse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", model_name)
        mlflow.log_param("version", "v1")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # ✅ Save model to a RELATIVE path
        local_model_path = os.path.join("temp_models", f"{model_name}.pkl")
        joblib.dump(model, local_model_path)

        # ✅ Log artifact using RELATIVE path (required on GitHub Actions)
        mlflow.log_artifact(local_model_path)

        # Track best model
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = model_name

# ✅ Register best model
with mlflow.start_run(run_name=f"{best_model_name}_Registration"):
    mlflow.log_param("best_model", best_model_name)
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="Best_BMI_Model"
    )
