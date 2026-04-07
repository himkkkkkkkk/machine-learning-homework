import pandas as pd

data_file = r"./krkopt.data"

data = pd.read_csv(data_file,sep=',',header=None)

print(data.head())

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder

y = y.apply(lambda x:0 if x=="draw" else 1)
y = y.values

label_encoder = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = label_encoder.fit_transform(X[col])

print(f"X is{1}",X.head())
print(f'y is {1}',y[:5])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scale = scaler.fit_transform(X)

from libsvm.svmutil import *
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scale,y,test_size=0.2)

C = 10000
gamma = 0.02

prob = svm_problem(y_train,X_train)
param = svm_parameter('-t 2 -c ' + str(C) + ' -g ' + str(gamma))

model = svm_train(prob,param)

y_pred,p_acc,p_val = svm_predict(y_test,X_test,model)

print("===== 模型信息 =====")
print(model)

print("\n支持向量个数:", model.get_nr_sv())

print("\n截距 rho:", model.rho[0])

print("\n前5个支持向量:")
print(model.get_SV()[:5])

print("训练样本数:", len(X_train))
print("支持向量数:", model.get_nr_sv())
