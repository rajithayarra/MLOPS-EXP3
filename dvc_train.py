import json
import os
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/iris.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"],
)

os.makedirs("pipeline_models", exist_ok=True)

for name in ["v1", "v2"]:
    p = params["models"][name]
    model = RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        random_state=params["train"]["random_state"],
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 5),
        "precision": round(precision_score(y_test, pred, average="weighted"), 5),
        "recall": round(recall_score(y_test, pred, average="weighted"), 5),
        "f1_score": round(f1_score(y_test, pred, average="weighted"), 5),
    }

    joblib.dump(model, f"pipeline_models/random_forest_{name}.pkl")
    with open(f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model {name}:", metrics)

print("Pipeline training completed successfully.")