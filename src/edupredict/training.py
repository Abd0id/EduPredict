import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib

csv_path = "../data/processed/edu-predict-clean.csv"

df = pd.read_csv(csv_path)

RANDOM_STATE = 42


target_col = "Exam_Score"
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)




lr_pipe = make_pipeline(StandardScaler(), LinearRegression())
lr_grid = GridSearchCV(
    lr_pipe,
    {
        "linearregression__fit_intercept":[True, False],
        "linearregression__positive":[True, False],
    },
    cv=kf, scoring="r2", n_jobs=-1,
)
lr_grid.fit(X_train, y_train)

rf_pipe = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=RANDOM_STATE))
rf_grid = GridSearchCV(
    rf_pipe,
    {
        "randomforestregressor__n_estimators": [100, 200, 400],
        "randomforestregressor__max_depth": [None, 5, 10, 20],
        "randomforestregressor__min_samples_split": [2, 5, 10],
    },
    cv=kf, scoring="r2", n_jobs=-1,
)
rf_grid.fit(X_train, y_train)

svr_pipe = make_pipeline(StandardScaler(), SVR())
svr_grid = GridSearchCV(
    svr_pipe,
    {
        "svr__kernel": ["rbf", "linear", "poly"],
        "svr__C": [0.1, 1, 10, 100],
        "svr__epsilon": [0.01, 0.1, 0.2, 0.5],
    },
    cv=kf, scoring="r2", n_jobs=-1,
)
svr_grid.fit(X_train, y_train)

models = {"RF": rf_grid, "SVR": svr_grid, "LR" : lr_grid}

for name, grid in models.items():
    y_pred = grid.predict(X_test)
    print(f"{name} | best CV R2: {grid.best_score_:.4f} | test R2: {r2_score(y_test, y_pred):.4f} | params: {grid.best_params_}")


best_name = max(models, key=lambda n: models[n].best_score_)
best_pipeline = models[best_name].best_estimator_

joblib.dump(best_pipeline, "../model/model_pipeline.pkl")
joblib.dump(list(X_train.coluns), "../model/expected_columns.pkl")