import pandas as pd
import numpy as np
import scipy
from matplotlib import cm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB



# find the combination that has the best accuracy
def find_best(X,y,scalers,encoders,models,kfold):
    accuracy=[]
    combination_case = []
    for scaler in range(len(scalers)):
        X=scaling(scalers[scaler][1],X)
        for model in range(len(models)):
            for k in kfold:
                hyperparameter_tuning(X,y,models[model],k,accuracy)
                combination_case.append([scalers[scaler][0],models[model][0],k])

    # show all the combination cases
    print('====combination cases====')
    for i in range(len(combination_case)):
        print(i+1,':',combination_case[i])
    # show all the accuracy
    print('====accuracy====')
    for i in range(len(accuracy)):
        print(i+1,':',accuracy[i])
    # show the best combination
    print('<Best Combination>')
    print('accuracy: ',max(accuracy),' scaler: ',combination_case[accuracy.index(max(accuracy))][0],' model: ',combination_case[accuracy.index(max(accuracy))][1],' K: ',combination_case[accuracy.index(max(accuracy))][2])

def hyperparameter_tuning(X,y,model,k,accuracy):
    print(X)
    if model[0]=='decision':
        #parameters
        param={
            'criterion': ['gini', 'entropy'],
            'max_depth': range(1, 10),
            'min_samples_split': range(2, 10),
            'min_samples_leaf': range(1, 5)
        }
    elif model[0]=='logistic_regression':
        #parameters
        param={
            'penalty':['none','l2'],
            'C':np.logspace(0,4,10)
        }
    elif model[0]=='svm':
        #parameters
        param={
            'C': scipy.stats.expon(scale=100),
            'gamma': scipy.stats.expon(scale=.1),
            'kernel': ['rbf'],
            'class_weight':['balanced', None]
        }



    cv = KFold(n_splits=k,shuffle=True,random_state=1)

    for train_ix,test_ix in cv.split(X):
        X_train,X_test=X.iloc[train_ix],X.iloc[test_ix]
        y_train,y_test=y.iloc[train_ix],y.iloc[test_ix]

        random = RandomizedSearchCV(estimator=model[1], param_distributions=param, n_iter=50, cv=3, verbose=2,
                                    random_state=42, n_jobs=-1)
        result=random.fit(X_train,y_train)

        # get the best performing model if on the whole training set
        best_model=result.best_estimator_
        # evaluate model on the hold out dataset
        yhat=best_model.predict(X_test)
        # evaluate the model
        acc=accuracy_score(y_test,yhat)
        # store accuracy
        accuracy.append(acc)

def scaling(scaler,X):
    scaled=scaler.fit_transform(X)
    scaled=pd.DataFrame(scaled)

    return scaled

#read data
df=pd.read_csv('breast-cancer-wisconsin.data',names=['ID','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelia Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Classes'])
print(df)

# ? to nan
df.replace({'?':np.nan},inplace=True)

# check the number of NaN
print('<Before data preprocessing-Nan>')
print(df.isna().sum())

# fill the Nan using bfill
df.fillna(method='bfill',inplace=True)

# check the number of NaN
print('<After data preprocessing-Nan>')
print(df.isna().sum())

# split data to features and target
X=df.drop(columns='Classes') # features
X=X.drop(columns='ID')
y=df['Classes'] # target

# count the number of each class
print(df['Classes'].value_counts()) # data imbalance

# Resolving the data imbalance
smote = SMOTE(random_state=0)
X_resampled,y_resampled=smote.fit_resample(X,y)
print("After OverSampling, counts of label '2': {}".format(sum(y_resampled==2)))
print("After OverSampling, counts of label '4': {}".format(sum(y_resampled==4)))

# scalers,encoders,models
scalers=[['standard',preprocessing.StandardScaler()],['minmax',preprocessing.MinMaxScaler()],['robust',preprocessing.RobustScaler()],['maxabs',preprocessing.MaxAbsScaler()]]
encoders=[['ordinal',preprocessing.OrdinalEncoder()],['onehot',preprocessing.OneHotEncoder()]]
models=[['decision',DecisionTreeClassifier()],['logistic_regression',LogisticRegression()],['svm',SVC()]]
kfold=[5,10,15,20]

# get the best
find_best(X,y,scalers,encoders,models,kfold)