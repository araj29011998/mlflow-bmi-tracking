import mlflow
import pandas as pd

# Load the test data
test_data = pd.read_csv("data/test.csv")

# Drop the actual BMI if present (we simulate "unseen" input)
X_test = test_data.drop(columns=["BMI"], errors="ignore")

# Load the model from the registry (production stage)
logged_model_uri = "models:/Best_BMI_Model/Production"
model = mlflow.sklearn.load_model(logged_model_uri)

# Make predictions
predictions = model.predict(X_test)

# Show predictions
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Predicted BMI = {pred:.2f}")
