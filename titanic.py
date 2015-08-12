__author__ = 'xymeow'

import pandas
import re
import numpy
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation

titanic = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")

predictors = ["Pclass", "Sex", "Fare", "Title", "Age", "SibSp", "Parch", "Embarked", "FamilySize", "FamilyId"]

title_maping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev":
    6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady":
                    10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
family_id_mapping = {}

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
     ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1),
     ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=8, min_samples_leaf=4),
     ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]]
]


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


def get_family_id(row):
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]


def set_data(dataset, title_maping):
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())
    dataset.loc[dataset["Sex"] == "male", "Sex"] = 0
    dataset.loc[dataset["Sex"] == "female", "Sex"] = 1
    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    dataset.loc[dataset["Embarked"] == "S", "Embarked"] = 0
    dataset.loc[dataset["Embarked"] == "C", "Embarked"] = 1
    dataset.loc[dataset["Embarked"] == "Q", "Embarked"] = 2
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"]
    dataset["NameLength"] = dataset["Name"].apply(lambda x: len(x))
    titles = titanic["Name"].apply(get_title)
    for k, v in title_maping.items():
        titles[titles == k] = v
    dataset["Title"] = titles
    family_ids = dataset.apply(get_family_id, axis=1)
    family_ids[dataset["FamilySize"] < 3] = -1
    dataset["FamilyId"] = family_ids



def predict(algorithms, train, test):
    full_predictions = []
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors], train["Survived"])
        predictions = alg.predict_proba(train[predictors].astype(float))[:, 1]
        full_predictions.append(predictions)
        # scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)
        # print(scores.mean())
    predictions = (full_predictions[0] * 3 + full_predictions[1] + full_predictions[2] * 0) / 4 #各算法权重
    predictions[predictions <= .5] = 0
    predictions[predictions > .5] = 1
    predictions = predictions.astype(int)
    return predictions


set_data(titanic, title_maping)
set_data(test, title_maping)
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

predictions = predict(algorithms, titanic, test)

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy) #for test

submission = pandas.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)
