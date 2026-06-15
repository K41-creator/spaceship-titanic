from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




pd.set_option('display.max_columns', None)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



all_data = pd.concat(
                     [train.drop(["Transported", "PassengerId", "Name"], axis=1), test.drop(["PassengerId", "Name"], axis=1)],
                     axis=0
                     )







cat_features = [
    "HomePlanet",
    "CryoSleep",
    "Cabin",
    "Destination",
    "VIP"
]







x = all_data.iloc[:len(train[cat_features]),:]
x_test = all_data.iloc[len(train[cat_features]):,:]

y = train['Transported']



print(x[cat_features].isnull().sum())



for col in cat_features:
    x[col] = x[col].fillna("Unknown").astype(str)
    x_test[col] = x_test[col].fillna("Unknown").astype(str)




model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=100,
    random_seed=42
)
model.fit(
    x,
    y,
    cat_features=cat_features
    )
pred = model.predict(x_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": pred
})

submission.to_csv(
    "submission.csv",
    index=False
)

"""
##検証

from sklearn.metrics import accuracy_score

score = accuracy_score(
    y,
    pred
)

print(score)


importance = model.get_feature_importance()

for col, imp in zip(
    x.columns,
    importance
):
    print(col, imp)

    """