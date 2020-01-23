import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
matplotlib.style.use('ggplot')
plt.figure(figsize=(9,9))
def sigmoid(t):
    return (1/(1 + np.e**(-t)))
plot_range = np.arange(-6, 6, 0.1)
y_values = sigmoid(plot_range)
plt.plot(plot_range,y_values, color="red")
titanic_train = pd.read_csv("../RICFiles/titanic_train.csv") 
char_cabin = titanic_train["Cabin"].astype(str) 
new_Cabin = np.array([cabin[0] for cabin in char_cabin])
titanic_train["Cabin"] = pd.Categorical(new_Cabin)
new_age_var = np.where(titanic_train["Age"].isnull(),28, titanic_train["Age"]) # Value if check is false
titanic_train["Age"] = new_age_var
label_encoder = preprocessing.LabelEncoder()
encoded_sex = label_encoder.fit_transform(titanic_train["Sex"])
log_model = linear_model.LogisticRegression()
log_model.fit(X = pd.DataFrame(encoded_sex),y = titanic_train["Survived"])
print(log_model.intercept_)
print(log_model.coef_)
preds = log_model.predict_proba(X= pd.DataFrame(encoded_sex))
preds = pd.DataFrame(preds)
preds.columns = ["Death_prob", "Survival_prob"]
pd.crosstab(titanic_train["Sex"], preds.loc[:, "Survival_prob"])
encoded_class = label_encoder.fit_transform(titanic_train["Pclass"])
encoded_cabin = label_encoder.fit_transform(titanic_train["Cabin"])
train_features = pd.DataFrame([encoded_class,encoded_cabin,encoded_sex, titanic_train["Age"]]).T
log_model = linear_model.LogisticRegression()
log_model.fit(X = train_features , y = titanic_train["Survived"])
print(log_model.intercept_)
print(log_model.coef_)
preds = log_model.predict(X= train_features)
pd.crosstab(preds,titanic_train["Survived"])
log_model.score(X = train_features ,y = titanic_train["Survived"])
metrics.confusion_matrix(y_true=titanic_train["Survived"],y_pred=preds)
print(metrics.classification_report(y_true=titanic_train["Survived"],y_pred=preds) )
titanic_test = pd.read_csv("../RICFiles/titanic_test.csv")
char_cabin = titanic_test["Cabin"].astype(str)
new_Cabin = np.array([cabin[0] for cabin in char_cabin])
titanic_test["Cabin"] = pd.Categorical(new_Cabin)
new_age_var = np.where(titanic_test["Age"].isnull(), 28,titanic_test["Age"])
titanic_test["Age"] = new_age_var
encoded_sex = label_encoder.fit_transform(titanic_test["Sex"])
encoded_class = label_encoder.fit_transform(titanic_test["Pclass"])
encoded_cabin = label_encoder.fit_transform(titanic_test["Cabin"])
test_features = pd.DataFrame([encoded_class,encoded_cabin,encoded_sex,titanic_test["Age"]]).T
test_preds = log_model.predict(X=test_features)
submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],"Survived":test_preds})
submission.to_csv("../RICFiles/tutorial_logreg_submission.csv",index=False)
print(pd)