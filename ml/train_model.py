import pandas as pd
import mlflow
import mlflow.sklearn
import joblib


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/match_features.csv")

X = df.drop("possession", axis=1)
y = df["possession"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Football_Boosting_Model")

with mlflow.start_run():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "boosting_model")

joblib.dump(model, "models/possession_model.pkl")
print("Model trained successfully")