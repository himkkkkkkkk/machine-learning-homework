import os

from pandas import read_csv

data_file = os.path.join(os.path.dirname(__file__), "krkopt.data")
data = read_csv(data_file, sep=",", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.preprocessing import LabelEncoder

y = y.apply(lambda x: 0 if x == "draw" else 1)
y = y.values

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scale = scaler.fit_transform(X)

# from libsvm import *  ai表示这个无法兼容sklearn，需要直接调用skl的api
# from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.3)

result = {}

for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    model = SVC(kernel=kernel, probability=True)
    param_grid = (
        {
            "C": [0.1, 1, 5],
            # 'kernel' : ['linear','poly','rbf','sigmoid'],
            "gamma": ["scale", "auto"],
        }
        if kernel != "poly"
        else {"C": [0.1, 1, 5], "gamma": ["scale", "auto"], "degree": [2, 3]}
    )

    gs = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)

    result[kernel] = {
        "best_params": gs.best_params_,
        "best_estimator": gs.best_estimator_,
    }

from ems import output

output(X_test, y_test, result)
# kf = KFold()
# for fold,(train_idx,test_idx) in enumerate(kf.split(X,y)):
#     X_train,X_test = X[train_idx],X[test_idx]
#     y_train,y_test = y[train_idx],y[test_idx]
# 貌似gs自动实现了交叉验证
