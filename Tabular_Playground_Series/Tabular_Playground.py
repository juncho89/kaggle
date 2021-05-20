import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# data import
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
submission = pd.read_csv('Data/sample_submission.csv')

feature = []
feature.append("id")
for i in range(50):
    feature.append("feature_%d"%i)


label = ['target']

x = train[feature]
y = train[label]

# Create Label Encoder
encoder = LabelEncoder()
result = encoder.fit_transform(y)
y = result
train['target'] = result

# inverse
encoder.inverse_transform(y)

import matplotlib.pyplot as plt
vc = train['target'].value_counts()
plt.pie(vc.values, labels = vc.index, autopct = '%.1f%%')
plt.show()

# correclation matrix
import seaborn as sns
corrmat = train.corr()
plt.figure(figsize=(52,52))
sns.heatmap(data = corrmat, annot=True,
fmt = '.2f', linewidths=.5, cmap='Blues')
plt.show()

# split test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y, random_state=34)

# LR
from sklearn.linear_model import LinearRegression
line_fitter = LinearRegression()
line_fitter.fit(x_train, y_train)
y_pred = line_fitter.predict(x_test)

from sklearn.linear_model import LinearRegression        ## 선형 회귀분석
from sklearn.linear_model import LogisticRegression      ## 로지스틱 회귀분석
from sklearn.naive_bayes import GaussianNB               ## 나이브 베이즈
from sklearn import svm                                  ## 서포트 벡터 머신
from sklearn import tree                                 ## 의사결정나무
from sklearn.ensemble import RandomForestClassifier      ## 랜덤포레스트

clf = LogisticRegression(solver='lbfgs').fit(x_train,y_train)
clf.predict(x_test)
clf.predict_proba(x_test)
clf.score(x_test,y_test)


clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
clf.predict_proba(x_test)
clf.score(y_pred, y_test)


prob = clf.predict_proba(test)
df = pd.DataFrame(data=prob)
submission['Class_1'] = df[0]
submission['Class_2'] = df[1]
submission['Class_3'] = df[2]
submission['Class_4'] = df[3]

submission.to_csv('submission.csv', index=False)

import xgboost as xgb
xgb_clf = xgb.XGBClassifier(random_state=42,tree_method='gpu_hist',subsample=0.7)
xgb_clf.fit(x_train, y_train, verbose=True)

from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss

lgbm_clf = LGBMClassifier()
lgbm_clf.fit(x_train, y_train)
y_pred = lgbm_clf.predict_proba(x_test)
log_loss(y_test, y_pred)




prob = lgbm_clf.predict_proba(test)
df = pd.DataFrame(data=prob)
submission['Class_1'] = df[0]
submission['Class_2'] = df[1]
submission['Class_3'] = df[2]
submission['Class_4'] = df[3]
submission.to_csv('submission_lgbm.csv', index=False)




clf.predict(x_test)
clf.predict_proba(x_test)
clf.score(x_test,y_test)




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)
lgbm_clf = LGBMClassifier()
lgbm_clf.fit(x_train, y_train)
y_pred = lgbm_clf.predict_proba(x_test)
log_loss(y_test, y_pred)



