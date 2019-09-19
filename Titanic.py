import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
sns.set(context="notebook", style="darkgrid", palette="deep", font="sans- serif", font_scale=0.7, color_codes=True)
##Importing data
Titanic_train = pd.read_csv("C:\\Users\\yashgoyal\\train.csv")

##Droping some columns which are not going to help in predictions

Titanic_train["familySize"] = Titanic_train["SibSp"]+Titanic_train["Parch"]+1
Titanic_1 = Titanic_train.drop(["Ticket","Name","Cabin","SibSp","Parch"],axis = 1)


table = pd.crosstab(Titanic_1["Survived"],Titanic_1["Sex"])
table

Titanic_1.groupby('Sex').Survived.mean()
Titanic_1.groupby('Pclass').Survived.mean()
Titanic_1.groupby(['Pclass','Sex']).mean()
Titanic_1.groupby(['Pclass','Sex']).mean()["Survived"].plot.bar()

def bar_chart(features):
    survived = Titanic_1[Titanic_1['Survived']==1][features].value_counts()
    Dead = Titanic_1[Titanic_1['Survived']==0][features].value_counts()
    df = pd.DataFrame([survived,Dead])
    df.index = ["survived","Dead"]
    df.plot(kind="bar",stacked=True,figsize=(10,5))
bar_chart("Sex")    
bar_chart("Pclass") 
bar_chart("Embarked") 
bar_chart("familySize")

facet = sns.FacetGrid(Titanic_1,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Age",shade=True)
facet.add_legend()

facet = sns.FacetGrid(Titanic_1,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"Pclass",shade=True)
facet.add_legend()

facet = sns.FacetGrid(Titanic_1,hue="Survived",aspect=4)
facet.map(sns.kdeplot,"familySize",shade=True)
facet.add_legend()

Pclass1 = Titanic_1[Titanic_1["Pclass"]==1]['Embarked'].value_counts()
Pclass2 = Titanic_1[Titanic_1["Pclass"]==2]['Embarked'].value_counts()
Pclass3 = Titanic_1[Titanic_1["Pclass"]==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ["1st class","2nd class","3rd class"]
df.plot(kind="bar",stacked=True)


##Visualizations
sns.countplot(x="Survived",data=Titanic_1)
sns.countplot(x="Survived",data=Titanic_1,hue="Sex")
sns.countplot(x="Survived",data=Titanic_1,hue="Pclass")
sns.countplot(x="Survived",data=Titanic_1,hue="Age")
sns.countplot(x="Age",data=Titanic_1,hue="Survived")
sns.countplot(x="Survived",data=Titanic_1,hue="Embarked")
sns.countplot(x="familySize",data=Titanic_1,hue="Survived")
sns.factorplot('Embarked',data=Titanic_1,hue='Pclass',kind='count')

sns.boxplot(x="Pclass",y="Age",data=Titanic)

FacetGrid = sns.FacetGrid(Titanic, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()

grid = sns.FacetGrid(Titanic, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

sns.barplot(x='Pclass', y='Survived', data=Titanic)

Titanic_1["Age"].plot.hist(color='red')
Titanic_1["Fare"].plot.hist(color='red',bins=40)

##EDA
Titanic_1.info()
Titanic_1.describe()
##Checking for null values
Titanic_1.isnull().sum()
## Checking null values by using heat map
sns.heatmap(Titanic_1.isnull(),cmap="viridis")
##Replacing null values by mean
Titanic_1["Age"].fillna(29.69,inplace=True)
##Replacing null values by mode
common_value = "S"
data = [Titanic_1]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
sns.heatmap(Titanic_1.isnull(),cmap="viridis")
## Coverting categorical columns in dummies
PC1 = pd.get_dummies(Titanic_1['Sex'],drop_first=True)
PC2 = pd.get_dummies(Titanic_1['Embarked'],drop_first=True)
PC3 = pd.get_dummies(Titanic_1['Pclass'],drop_first=True)
##Adding dummies columns in dataset
Titanic_1  = pd.concat([Titanic_1,PC1,PC2,PC3],axis=1)
## Droping columns which has been created into dummies 
Titanic_1  = Titanic_1.drop(["Sex","Embarked","Pclass"],axis=1)

## Implementing Logistic regression technique
X_train = Titanic_1.drop(["Survived"],axis=1) ##Input
y_train = Titanic_1["Survived"] ##Output

##### Now i will  convert test data #############
Titanic_test = pd.read_csv("C:\\Users\\yashgoyal\\test.csv")

Titanic_test["familySize"] = Titanic_test["SibSp"]+Titanic_test["Parch"]+1
Titanic_2 = Titanic_test.drop(["Ticket","Name","Cabin","SibSp","Parch"],axis = 1)

##EDA
Titanic_2.info()
Titanic_2.describe()
##Checking for null values
Titanic_2.isnull().sum()
## Checking null values by using heat map
sns.heatmap(Titanic_2.isnull(),cmap="viridis")
##Replacing null values by mean
Titanic_2["Age"].fillna(30.27,inplace=True)
Titanic_2["Fare"].fillna(35.62,inplace=True)
## Coverting categorical columns in dummies
PC4 = pd.get_dummies(Titanic_2['Sex'],drop_first=True)
PC5 = pd.get_dummies(Titanic_2['Embarked'],drop_first=True)
PC6 = pd.get_dummies(Titanic_2['Pclass'],drop_first=True)
##Adding dummies columns in dataset
Titanic_2  = pd.concat([Titanic_2,PC4,PC5,PC6],axis=1)
## Droping columns which has been created into dummies 
Titanic_2  = Titanic_2.drop(["Sex","Embarked","Pclass"],axis=1)

X_test = Titanic_2


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
pred_test = logmodel.predict(X_test)
pred_train = logmodel.predict(X_train)
accuracy_score(pred_train,y_train)  ###80.58
pro_of_test = pd.DataFrame(logmodel.predict_proba(X_test),columns=["Prob of didn't survive","survived"])
pro_of_test  ## Probabilities of survival

RE = pd.concat([X_test.PassengerId],axis=1)
RE['Survived'] = pred_test
RE.head()
RE.to_csv("Submission.csv",index=False)

model_log = logmodel
Adaboost_log =  AdaBoostClassifier(base_estimator=model_log ,n_estimators=400,learning_rate=1)
boostmodel_log =Adaboost_log.fit(X_train,y_train)
boost_pred_log = boostmodel_log.predict(X_test)
boost_pred_log_train = boostmodel_log.predict(X_train)
accuracy_score(boost_pred_log_train,y_train)   ### 80.35


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
## For training
classification_report(y_train,pred_train)
confusion_matrix(y_train,pred_train)
accuracy_score(y_train,pred_train)   ##80.58%
#ROC curve for test data
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_train, pred_train)
roc_auc = roc_auc_score(y_train, pred_train)
plt.figure()
plt.plot(fpr, tpr, label = 'Logistic Regression Sensitivity = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.title('ROC curve for train data')
plt.legend(loc="lower Right")
plt.show()



##Other classifications techniques  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10,shuffle=True,random_state=True)
##KNN
clf = KNeighborsClassifier(n_neighbors=13)
scoring = "accuracy"
score = cross_val_score(clf,X_train,y_train,cv = k_fold,n_jobs=1,scoring=scoring)
print(score)
round(np.mean(score)*100,2)  ##63.75%

from sklearn.ensemble import AdaBoostClassifier
##Decision tree
clf = DecisionTreeClassifier()
scoring = "accuracy"
score = cross_val_score(clf,X_train,y_train,cv = k_fold,n_jobs=1,scoring=scoring)
print(score)
round(np.mean(score)*100,2)   ##76.55%
model_D = DecisionTreeClassifier(criterion="entropy",max_depth=1)
Adaboost =  AdaBoostClassifier(base_estimator=model_D,n_estimators=400,learning_rate=1)
boostmodel = Adaboost.fit(X_train,y_train)
boost_pred = boostmodel.predict(X_test)
boost_pred_train = boostmodel.predict(X_train)
accuracy_score(boost_pred_train,y_train)   ### 87.65

##Random forest
clf = RandomForestClassifier(n_estimators=13)
scoring = "accuracy"
score = cross_val_score(clf,X_train,y_train,cv = k_fold,n_jobs=1,scoring=scoring)
print(score)
round(np.mean(score)*100,2)   ##81.38%
model_R = clf
Adaboost_R =  AdaBoostClassifier(base_estimator=model_R,n_estimators=400,learning_rate=1)
boostmodel_R = Adaboost_R.fit(X_train,y_train)
boost_pred_R = Adaboost_R.predict(X_test)
boost_pred_R_train = Adaboost_R.predict(X_train)
accuracy_score(boost_pred_R_train,y_train)   #### 100%

RER = pd.concat([X_test.PassengerId],axis=1)
RER['Survived'] = boost_pred_R
RER.head()

RER.to_csv("SubmissionBOOst.csv",index=False)
###### XGBOOST ##########
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
params = {"learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],"max_depth":[1,2,3,4,5,6,8,9,10],"min_child_weight":[1,2,3,4,5,6,7,8,9],"gamma":[0.0,0.1,0.2,0.3,0.4,0.5],"colsample_bytree":[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],"n_estimators":[100,200,300,400,500]}
classifier = XGBClassifier()
random_search = RandomizedSearchCV(classifier,param_distributions=params,n_iter=10,scoring="roc_auc",n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_train,y_train)
random_search.best_estimator_

XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.2,
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=None, n_estimators=200, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

XGB_fit = XGB.fit(X_train,y_train)
XGB_pred = XGB.predict(X_test)
XGB_pred_train = XGB.predict(X_train)
accuracy_score(XGB_pred_train,y_train)  #### 94.94%
XG = pd.concat([X_test.PassengerId],axis=1)
XG['Survived'] = XGB_pred
XG.to_csv("finalXGboost.csv",index=False)
##Naive Bayes
clf = GaussianNB()
scoring = "accuracy"
score = cross_val_score(clf,X_train,y_train,cv = k_fold,n_jobs=1,scoring=scoring)
print(score)
round(np.mean(score)*100,2)   ##78.12%

##SVM
clf = SVC()
scoring = "accuracy"
score = cross_val_score(clf,X_train,y_train,cv = k_fold,n_jobs=1,scoring=scoring)
print(score)
round(np.mean(score)*100,2)   ##60.94%



