import pandas as pd
import xgboost as xgb
from sklearn import tree, ensemble, svm, neighbors
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Optional


class Model:
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame, predict: pd.DataFrame):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target['target'], test_size=0.2)
        self.to_predict = predict
        self.model = None
        self.name: Optional[str] = None

    def fit(self):
        self.model.fit(self.x_train, self.y_train)
        prediction = self.model.predict(self.x_test)
        score = roc_auc_score(self.y_test, prediction)
        print("%s: %.2f%%" % (self.name, score * 100.0))

    def predict(self) -> pd.DataFrame:
        return self.model.predict_proba(self.to_predict)


class Xgboost(Model):
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame, predict: pd.DataFrame):
        super().__init__(data, target, predict)
        self.model = xgb.XGBClassifier()
        self.name = 'XGboost'


class DecisionTree(Model):
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame, predict: pd.DataFrame):
        super().__init__(data, target, predict)
        self.model = tree.DecisionTreeClassifier()
        self.name = 'Decision Tree'


class RandomForest(Model):
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame, predict: pd.DataFrame):
        super().__init__(data, target, predict)
        self.model = ensemble.RandomForestClassifier(n_estimators=70)
        self.name = 'Random Forest'


class SVM(Model):
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame, predict: pd.DataFrame):
        super().__init__(data, target, predict)
        self.model = svm.SVC()
        self.name = 'SVM'


class KNN(Model):
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame, predict: pd.DataFrame):
        super().__init__(data, target, predict)
        self.model = neighbors.KNeighborsClassifier()
        self.name = 'KNN'


class Ensamble(Model):
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame, predict: pd.DataFrame):
        super().__init__(data, target, predict)
        xgb1 = xgb.XGBClassifier(seed=113)
        xgb2 = xgb.XGBClassifier(seed=439)
        xgb3 = xgb.XGBClassifier(seed=677)

        self.model = VotingClassifier(estimators=[('xgb1', xgb1), ('xgb2', xgb2), ('xgb3', xgb3)], voting='soft')
        self.name = 'ensamble'
