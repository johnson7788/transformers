import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
# from imblearn.over_sampling import SMOTE
# from imblearn.combine import SMOTEENN
# from imblearn.combine import SMOTETomek
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

file = 'PAF_cacdmean_prpampmedian.csv'
data = pd.read_csv(file)
predictors = data.columns[1:]      #predictors是用于预测的特征的index
type2cl = data.columns[0]          #type2cl是前面的标签的index，在第一列
X = data[predictors]
Y = data.iloc[:, 0]

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

#切分数据
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

#建立模型，用Adaboost方法
model = AdaBoostClassifier(n_estimators=100, learning_rate=0.8)
model.fit(X_train, y_train)
# 类别的数量
print(model.n_classes_)
# 返回类别标签
print(model.classes_)
# 特征的权重
print(model.feature_importances_)

stat=pd.DataFrame(columns=['importance','feature'])
stat['importance']=model.feature_importances_
stat['feature']=X_train.columns
stat.sort_values(by='importance',ascending=False,inplace=True)
stat.to_excel('FeatureEvaluate.xls',index=False)
