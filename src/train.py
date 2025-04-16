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

# Model dictionary
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5),
    "RandomForest": RandomForestRegressor()
}

# Set experiment
mlflow.set_experiment("BMI_Prediction_Experiment")

# To track the best model
best_model_name = None
best_model = None
best_mae = float('inf')  # Lower is better

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log metrics
        mlflow.log_param("model", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("version", "v1")

        # Save locally
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Track best
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = model_name

# Register the best model
with mlflow.start_run(run_name=f"{best_model_name}_Registration") as run:
    mlflow.log_param("best_model", best_model_name)
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="Best_BMI_Model"
    )
