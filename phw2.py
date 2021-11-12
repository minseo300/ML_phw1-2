# Import Class Libraries
import eyeball as eyeball
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import purity as purity
import seaborn as sns
import sklearn
import plotly.express as px
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from scipy.stats import stats
from pyclustering.cluster.clarans import clarans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
sns.set()

# Import Class Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

sns.set()

from scipy.stats import stats
# from pyclustering.cluster.clarans import clarans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

from sklearn.metrics import silhouette_score
from sklearn import metrics


# Import Class Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import purity
import seaborn as sns
from scipy.stats import stats
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
#from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score, make_scorer

sns.set()

from scipy.stats import stats

#####################################################################
# Dataset = Online Shoppers Purchasing Intention
# Feature = Administrative, Administrative Duration, Informational,
#           Informational Duration, Product Related, Product Related Duration,
#           Bounce Rate, Exit Rate, Page Value, Special Day, Browser, Region,
#           Traffic Type, Visitor Type, Weekend, Operating Systems, Month
# Target  = Revenue

# Number of dataset = 12,330
# Numerical value   = Administrative, Administrative Duration, Informational,
#                     Informational Duration, Product Related, Product Related Duration,
#                     Bounce Rate, Exit Rate, Page Value, Special Day
# Categorical value = Browser, Region, Traffic Type, Visitor Type, Weekend,
#                     Operating Systems, Month, Revenue

df = pd.read_csv('online_shoppers_intention.csv')

print(list(df.columns.values))
feature_label = ['Administrative', 'Administrative_Duration', 'Informational',
                 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                 'BounceRates', 'ExitRate', 'PageValue', 'SpecialDay', 'Browser', 'Region',
                 'TrafficType', 'VisitorType', 'Weekend', 'OperatingSystems', 'Month']
target_label = ['Revenue']

scale_col = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
             'ProductRelated', 'ProductRelated_Duration', 'BounceRates',
             'ExitRates', 'PageValues', 'SpecialDay']
encode_col = ['Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'OperatingSystems', 'Month']

# Print dataset's information
# print("\n***** Online Shoppers Intention *****")
# print(df.head())

# print("\n************ Description *************")
# print(df.describe())

# print("\n************ Information *************")
# print(df.info())

# Check null value
# print("\n************ Check null *************")
# print(df.isna().sum())

# Fill null value - Using ffill
df = df.fillna(method='ffill')
# Check null value (Cleaned Data)
print("\n***** Check null (Cleaned Data) *****")
print(df.isna().sum())


# Remove Outliers with z-score
# Description = Use the z-score to handle outlier over mean +- 3SD
# Input  = dataframe's column
# Output = index
def find_outliers(col):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z > 3, True, False)
    return pd.Series(idx_outliers, index=col.index)


# Remove outliers (Numerical value)
for n in range(len(scale_col)):
    idx = None
    idx = find_outliers(df.iloc[:, n])
    df = df.loc[idx == False]

# print("\n****** Removed Outlier (Numerical value) *****")
# print(df.info())

# Remove outliers (Categorical value)
# print("\n***** Check outlier of categorical values *****")
# for n in encode_col:
#     print(df[n].value_counts(), "\n")

for n in [11, 12, 14]:
    idx = None
    idx = find_outliers(df.iloc[:, n])
    df = df.loc[idx == False]
df = df[df['VisitorType'] != 'Other']


# print("\n***** Removed Outlier (Categorical value) *****")
# for n in encode_col:
#     print(df[n].value_counts(), "\n")

# print("\n***** Cleaned Dataset *****")
# print(df.info())

# Scoring function
def overall_average_score(actual, prediction):
    precision, recall, f1_score, _ = precision_recall_fscore_support(actual, prediction, average='binary')
    total_score = matthews_corrcoef(actual, prediction) + accuracy_score(actual,
                                                                         prediction) + precision + recall + f1_score
    return total_score / 5


df.columns = df.columns.to_series().apply(lambda x: x.strip())
# Set X, y data
y_data = df.loc[:, target_label]
X_data = df.drop(target_label, axis=1)


def FindBestAccruacy(X, y, scale_col, encode_col, scalers=None, encoders=None,
                     models=None, model_param=None, cv=None, n_jobs=None):
    # Set Encoder
    if encoders is None:
        # encode = [OrdinalEncoder(), LabelEncoder()]
        encode=[LabelEncoder()]
    else:
        encode = encoders

    # Set Scaler
    if scalers is None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else:
        scale = scalers

    # Set Model
    if models is None:
        model = [
            LogisticRegression(),
            SVC(),
            GradientBoostingClassifier()
        ]
    else:
        model = models

    # Set Hyperparameter
    if model_param is None:

        parameter = [
            # LogisticRegression()
            {'penalty': ['l1', 'l2'], 'random_state': [0, 1], 'C': [0.01, 0.1, 1.0, 10.0, 100.0],
             'solver': ["lbfgs", "sag", "saga"], 'max_iter': [10, 50, 100]},
            # SVC()
            {'random_state': [0, 1], 'kernel': ['linear', 'rbf', 'sigmoid'],
             'C': [0.01, 0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto']},
            # GradientBoostingClassifier()
            {'loss': ['deviance', 'exponential'],
             'learning_rate': [0.001, 0.1, 1],
             'n_estimators': [1, 10, 100, 1000],
             'subsample': [0.0001, 0.001, 0.1],
             'min_samples_split': [10, 50, 100, 300],
             'min_samples_leaf': [5, 10, 15, 50]}
        ]

    else:
        parameter = model_param

    # Set CV(cross validation)
    if cv is None:
        setCV = 5
    else:
        setCV = cv

    # Set n_jobs
    if n_jobs is None:
        N_JOBS = -1
    else:
        N_JOBS = n_jobs

    best_score = 0
    best_combination = {}
    param = {}

    # SMOTE - Synthetic minority oversampling technique (Fixing the imbalanced data)
    target = y
    # smote = SMOTE(random_state=len(X))
    # X, y = smote.fit_resample(X, y)

    ####################################################################
    # Iterate
    for i in scale:
        for j in encode:
            # Scaling
            df_scaled = pd.DataFrame(i.fit_transform(X[scale_col]))
            df_scaled.columns = scale_col
            # Encoding
            if encode_col is not None:
                if type(j) == type(OrdinalEncoder()):
                    df_encoded = pd.DataFrame(j.fit_transform(X[encode_col]))
                    df_encoded.columns = encode_col
                    df_encoded.index = df_scaled.index
                    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)

                    # y=pd.DataFrame(j.fit_transform(y)) # todo
                else:
                    print("No")
                    dum = pd.DataFrame(pd.get_dummies(X[encode_col]))
                    print(dum.head())
                    dum.index = df_scaled.index
                    df_prepro = pd.concat([df_scaled, dum], axis=1)

                    # y=pd.DataFrame(pd.get_dummies(y)) # todo
            else:
                df_prepro = df_scaled[pd.get_dummies(y)]
            df_prepro = pd.DataFrame(df_prepro)
            # smote = SMOTE(random_state=len(df_prepro))
            # df_prepro, y = smote.fit_resample(df_prepro, y)

            # Feature Selection Using the Select KBest (K = 6)
            selectK = SelectKBest(score_func=f_regression, k=6).fit(df_prepro, y.values.ravel())
            cols = selectK.get_support(indices=True)
            df_selected = df_prepro.iloc[:, cols]
            df_selected = df_selected.fillna(method='ffill')

            for z in model:
                print("model: ", z)
                print(z.get_params().keys())
                # Split train, testset
                X_train, X_test, y_train, y_test = train_test_split(df_selected, y)

                # Set hyperparameter
                if model_param is None:
                    if model[0] is z:
                        param = parameter[0]
                    elif model[1] is z:
                        param = parameter[1]
                    elif model[2] is z:
                        param = parameter[2]
                    elif model[3] is z:
                        param = parameter[3]

                else:
                    param = parameter

                # grid_scorer = make_scorer(overall_average_score, greater_is_better=True)

                # Modeling(Using the RandomSearchCV)
                random_search = RandomizedSearchCV(estimator=z, param_distributions=param, n_jobs=N_JOBS, cv=setCV)
                random_search.fit(X_train, y_train.values.ravel())
                score = random_search.score(X_test, y_test)

                # Find Best Score
                if best_score == 0 or best_score < score:
                    best_score = score
                    best_combination['scaler'] = i
                    best_combination['encoder'] = j
                    best_combination['model'] = z
                    best_combination['parameter'] = random_search.best_params_

    # best_model = random_search.best_estimator_
    ##pred = best_model.predict(X_test)
    # precision, recall, f1_score, _ = precision_recall_fscore_support(y, pred, average='binary')
    # total_score = matthews_corrcoef(y, pred) + accuracy_score(y, pred) + precision + recall + f1_score
    # print("precision: {} recall: {} f1_score: {}".format(precision, recall, f1_score))
    # print("total avg score: {}".format(total_score / 5)) #todo

    # Print them
    print("Best Score = {:0.6f}".format(best_score), "")
    print("Best Combination, Model {}, Encoder {}, Scaler {}".
          format(best_combination['model'], best_combination['encoder'], best_combination['scaler']))
    print("Hyperparameter {}".format(best_combination['parameter']))

    return


# Auto Find Best Accuracy
print("Auto Find Best Accuracy")
# FindBestAccruacy(X_data, y_data, scale_col=scale_col, encode_col=encode_col,models=None, model_param=None )
FindBestAccruacy(X_data, y_data, scale_col=scale_col, encode_col=encode_col, encoders=None, scalers=None, models=None,
                 model_param=None)

   
# Import Class Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import purity
import seaborn as sns
from scipy.stats import stats
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score, make_scorer

sns.set()

from scipy.stats import stats

#####################################################################
# Dataset = Online Shoppers Purchasing Intention
# Feature = Administrative, Administrative Duration, Informational,
#           Informational Duration, Product Related, Product Related Duration,
#           Bounce Rate, Exit Rate, Page Value, Special Day, Browser, Region,
#           Traffic Type, Visitor Type, Weekend, Operating Systems, Month
# Target  = Revenue

# Number of dataset = 12,330
# Numerical value   = Administrative, Administrative Duration, Informational,
#                     Informational Duration, Product Related, Product Related Duration,
#                     Bounce Rate, Exit Rate, Page Value, Special Day
# Categorical value = Browser, Region, Traffic Type, Visitor Type, Weekend,
#                     Operating Systems, Month, Revenue

df = pd.read_csv('online_shoppers_intention.csv')

print(list(df.columns.values))
feature_label = ['Administrative', 'Administrative_Duration', 'Informational',
                 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                 'BounceRates', 'ExitRate', 'PageValue', 'SpecialDay', 'Browser', 'Region',
                 'TrafficType', 'VisitorType', 'Weekend', 'OperatingSystems', 'Month']
target_label = ['Revenue']

scale_col = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
             'ProductRelated', 'ProductRelated_Duration', 'BounceRates',
             'ExitRates', 'PageValues', 'SpecialDay']
encode_col = ['Browser', 'Region', 'TrafficType', 'VisitorType','Weekend', 'OperatingSystems', 'Month']


# Print dataset's information
# print("\n***** Online Shoppers Intention *****")
# print(df.head())

# print("\n************ Description *************")
# print(df.describe())

# print("\n************ Information *************")
# print(df.info())

# Check null value
# print("\n************ Check null *************")
# print(df.isna().sum())

# Fill null value - Using ffill
df = df.fillna(method='ffill')
# Check null value (Cleaned Data)
print("\n***** Check null (Cleaned Data) *****")
print(df.isna().sum())

# Remove Outliers with z-score
# Description = Use the z-score to handle outlier over mean +- 3SD
# Input  = dataframe's column
# Output = index
def find_outliers(col):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z > 3, True, False)
    return pd.Series(idx_outliers, index=col.index)

# Remove outliers (Numerical value)
for n in range(len(scale_col)):
    idx = None
    idx = find_outliers(df.iloc[:, n])
    df = df.loc[idx == False]

# print("\n****** Removed Outlier (Numerical value) *****")
# print(df.info())

# Remove outliers (Categorical value)
# print("\n***** Check outlier of categorical values *****")
# for n in encode_col:
#     print(df[n].value_counts(), "\n")

for n in [11, 12, 14]:
    idx = None
    idx = find_outliers(df.iloc[:, n])
    df = df.loc[idx == False]
df = df[df['VisitorType'] != 'Other']

# print("\n***** Removed Outlier (Categorical value) *****")
# for n in encode_col:
#     print(df[n].value_counts(), "\n")

# print("\n***** Cleaned Dataset *****")
# print(df.info())

#Scoring function
def overall_average_score(actual,prediction):
    precision, recall, f1_score, _ = precision_recall_fscore_support(actual, prediction, average='binary')
    total_score = matthews_corrcoef(actual, prediction)+accuracy_score(actual, prediction)+precision+recall+f1_score
    return total_score/5
df.columns = df.columns.to_series().apply(lambda x: x.strip())
# Set X, y data
y_data = df.loc[:, target_label]
X_data = df.drop(target_label, axis=1)

def FindBestAccruacy(X, y, scale_col, encode_col, scalers=None, encoders=None,
                      models=None, model_param=None, cv=None, n_jobs=None):

    # Set Encoder
    if encoders is None:
        encode = [OrdinalEncoder(), OneHotEncoder(), LabelEncoder()]
    else: encode = encoders

    # Set Scaler
    if scalers is None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: scale = scalers

    # Set Model
    if models is None:
        model = [
            LogisticRegression(),
            SVC(),
            GradientBoostingClassifier()
        ]
    else: model = models

    # Set Hyperparameter
    if model_param is None:
                    
        parameter = [
                      # LogisticRegression()
                     {'penalty':['l1','l2'], 'random_state':[0,1], 'C':[0.01, 0.1, 1.0, 10.0, 100.0],
                       'solver':["lbfgs", "sag", "saga"], 'max_iter':[10, 50, 100]},
                      # SVC()
                     {'random_state': [0,1], 'kernel': ['linear', 'rbf', 'sigmoid'],
                       'C': [0.01, 0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto']},
                    # GradientBoostingClassifier()
                     {'loss':['deviance','exponential'],
                      'learning_rate':[0.001, 0.1, 1],
                      'n_estimators':[1, 10,100,1000],
                      'subsample':[0.0001,0.001, 0.1],
                      'min_samples_split':[10,50, 100, 300],
                      'min_samples_leaf':[5, 10, 15,50]}
                     ]

    else: parameter = model_param

    # Set CV(cross validation)
    if cv is None:
        setCV = 5
    else: setCV = cv

    # Set n_jobs
    if n_jobs is None:
        N_JOBS = -1
    else: N_JOBS = n_jobs

    best_score = 0
    best_combination = {}
    param = {}

    # SMOTE - Synthetic minority oversampling technique (Fixing the imbalanced data)
    target = y
    # smote = SMOTE(random_state=len(X))
    # X, y = smote.fit_resample(X, y)

    ####################################################################
    # Iterate
    for i in scale:
        for j in encode:
            # Scaling
            df_scaled = pd.DataFrame(i.fit_transform(X[scale_col]))
            df_scaled.columns = scale_col
            # Encoding
            if encode_col is not None:
                if type(j) == type(OrdinalEncoder()):
                    df_encoded = pd.DataFrame(j.fit_transform(X[encode_col]))
                    df_encoded.columns = encode_col
                    df_encoded.index = df_scaled.index
                    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)

                    #y=pd.DataFrame(j.fit_transform(y)) # todo
                else:
                    print("No")
                    dum = pd.DataFrame(pd.get_dummies(X[encode_col]))
                    dum.index = df_scaled.index
                    df_prepro = pd.concat([df_scaled, dum], axis=1)

                    #y=pd.DataFrame(pd.get_dummies(y)) # todo
            else:
                df_prepro = df_scaled[pd.get_dummies(y)]
            df_prepro = pd.DataFrame(df_prepro)
            #smote = SMOTE(random_state=len(df_prepro))
            #df_prepro, y = smote.fit_resample(df_prepro, y)


            # Feature Selection Using the Select KBest (K = 6)
            selectK = SelectKBest(score_func=f_regression, k=6).fit(df_prepro, y.values.ravel())
            cols = selectK.get_support(indices=True)
            df_selected = df_prepro.iloc[:, cols]
            df_selected = df_selected.fillna(method='ffill')

            for z in model:
                print("model: ",z)
                print(z.get_params().keys())
                # Split train, testset
                X_train, X_test, y_train, y_test = train_test_split(df_selected, y)

                # Set hyperparameter
                if model_param is None:
                    if model[0] is z:
                        param = parameter[0]
                    elif model[1] is z:
                        param = parameter[1]
                    elif model[2] is z:
                        param = parameter[2]
                    elif model[3] is z:
                        param = parameter[3]

                else: param = parameter



                #grid_scorer = make_scorer(overall_average_score, greater_is_better=True)

                # Modeling(Using the RandomSearchCV)
                random_search = RandomizedSearchCV(estimator=z, param_distributions=param, n_jobs=N_JOBS,  cv=setCV)
                random_search.fit(X_train, y_train.values.ravel())
                score = random_search.score(X_test, y_test)


                # Find Best Score
                if best_score == 0 or best_score < score:
                    best_score = score
                    best_combination['scaler'] = i
                    best_combination['encoder'] = j
                    best_combination['model'] = z
                    best_combination['parameter'] = random_search.best_params_

    #best_model = random_search.best_estimator_
    ##pred = best_model.predict(X_test)
    #precision, recall, f1_score, _ = precision_recall_fscore_support(y, pred, average='binary')
    #total_score = matthews_corrcoef(y, pred) + accuracy_score(y, pred) + precision + recall + f1_score
    #print("precision: {} recall: {} f1_score: {}".format(precision, recall, f1_score))
    #print("total avg score: {}".format(total_score / 5)) #todo
    
    
    # Print them
    print("Best Score = {:0.6f}".format(best_score), "")
    print("Best Combination, Model {}, Encoder {}, Scaler {}".
          format(best_combination['model'], best_combination['encoder'], best_combination['scaler']))
    print("Hyperparameter {}".format(best_combination['parameter']))

    return


# Auto Find Best Accuracy
print("Auto Find Best Accuracy")
#FindBestAccruacy(X_data, y_data, scale_col=scale_col, encode_col=encode_col,models=None, model_param=None )
FindBestAccruacy(X_data, y_data, scale_col=scale_col, encode_col=encode_col,encoders = None, scalers = None,models=None, model_param=None )


#
# ##############################################################################
# # AutoML (X, y = None, scale_col, encode_col, scalers = None, encoders = None,
# #         feature_param = None, models = None, model_param = None,
# #         scores = None, score_param = None)
# #
# # **************************************************************************************
# # ******************************** Must Read *******************************************
# # **************************************************************************************
# # Description = When parameters are put in, the plot and scores are output
# #               The method of producing results in AutoML function consists of three main steps
# #
# #           Step 1 = Feature Selection (PCA(), RandomSelect(), CustomSelect()) * model (KMeans(), GMM(), clarans(), DBSCAN(), OPTICS()) = 15,
# #                    Find a combination with the best silhouette score in each combination
# #
# #           Step 2 = If there is a target value, Among the three Feature Selection (PCA(), RandomSelect(), CustomSelect()),
# #                    check which model has the highest purity and return three results
# #
# #           Step 3 = Using the final three combinations (without a target value),
# #                    we compare with the combinations (with a target value)
# #               - The results are checked through the clustering plot and the silhouette score -
# # ***************************************************************************************
# # ***************************************************************************************
# #
# # Input = X: Data Feature
# #         Y: Data Target (If you have a target value, enter it)
# #         Scale_col: columns to scaled
# #         Encode_col: columns to encode
# #         Scalers: list of scalers
# # 	         None: [StandardScaler(), RobustScaler(), MinMaxScaler()]
# #            If you want to scale other ways, then put the scaler in list.
# #         Encoders: list of encoders
# # 	         None: [OrdinalEncoder(), LabelEncoder()]
# # 	         If you want to encode other ways, then put the encoder in list.
# #
# #         Feature: list of features
# #            None: [PCA(), RandomSelect(), CustomSelect()]
# #            If you want to set other ways, then put specific feature in list
# #
# #         Feature_param: feature selection method's parameter
# #            PCA()'s None: [n_components: None (int)]
# #            RandomSelect()'s None: [number_of_features: None (int)]
# #            CustomSelect()'s None: [combination_of_features: None (list)]
# #
# #         Models: list of models
# #            None: [KMeans(), GMM(), clarans(), DBSCAN(), OPTICS()]
# #            If you want to fit other ways, then put (Clustering)model in list.
# #
# #         Model_param: list of model's hyperparameter
# #         KMeans()’s None: [n_clusters: None (int), init: None(k-means++, random),
# #                           n_init: None (int), Random_state: None (int), max_iter: None (int)]
# #         GMM()’s None: [n_components: None (int), covariance_type: None (spherical, tied, diag),
# #                        n_init: None (int), Random_state: None (int),
# #                        min_covar: None (float), tol: None (float)]
# #         clarans()’s None: [number_clusters: None (int), numlocal_minima: None (int),
# #                            max_neighbor: None (int)]
# #         DBSCAN()’s None: [eps: None (float), min_samples: None (int), metric: None (str or callable),
# #                           p: None (float), Algorithm: None (auto, ball_tree, kd_tree, brute)]
# #         OPTICS()’s None: [eps: None (float), min_samples: None (int), p: None (int),
# #                           cluster_method: None (xi, dbscan), algorithm: None (auto, ball_tree, kd_tree, brute)]
# #         If you want to set other ways, then put the hyperparameter in list
# #
# #         Scores: list of score methods
# #            None: [silhouette_score(), KelbowVisualizer(), purity(), eyeball()]
# #            If you want to see other ways, then put the scoring model in list.
# #
# #         Score_param: list of score method's hyperparameter
# #      		 Silhouette_score()’s None: [metric: None (str, callable), random_state: None (int)]
# #            Purity()’s None: None
# #            eyeball()'s None: None
# #
# # Output = some scores, plots
#
# # Description = Calculate the silhouette score and return the value
# # Input  = kind of model, Dataset
# # Output = Silhouette score
# def cv_silhouette_scorer(estimator, X):
#     print("그리드 서치중 : ", estimator)
#
#     # If GMM(EM) handle separately
#     if type(estimator) is sklearn.mixture._gaussian_mixture.GaussianMixture:
#         # print("it's GaussianMixture()")
#         labels = estimator.fit_predict(X)
#         return silhouette_score(X, labels, metric='euclidean')
#
#     # Calculate and return Silhouette score
#     else:
#         estimator.fit(X)
#         cluster_labels = estimator.labels_
#         num_labels = len(set(cluster_labels))
#         num_samples = len(X.index)
#         if num_labels == 1 or num_labels == num_samples:
#             return -1
#         else:
#             return silhouette_score(X, cluster_labels)
#
#
# # purity를 구해주는 함수
# def purity_score(y_true, y_pred):
#     # compute contingency matrix
#     contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
#     # return purity
#     return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
#
#
# # Description = randomly determines features
# # input  = Dataset, number of feature
# # Output = (Random)Dataset
# class RandomSelect:
#     # number of feature (Default:4)
#     n = 4
#
#     # Accept N
#     def set_params(self, n_components):
#         self.n = n_components
#
#     # Pick N and combination
#     def fit_transform(self, data):
#         choice = np.random.choice(data.columns, self.n)
#         result = pd.DataFrame(data[choice[0]])
#
#         for i in range(1, len(choice)):
#             result = pd.concat([result, data[choice[i]]], axis=1)
#
#         # Return Dataset
#         return result
#
#
# # Description = select specific features
# # input  = Dataset, selected features
# # Output = (Selected) Dataset
# class CustomSelect:
#     # Combination of selected features
#     feature = None
#
#     # Accept selected features
#     def set_params(self, n_components):
#         self.feature = n_components
#
#     # Combine the selected features
#     def fit_transform(self, data):
#         result = pd.DataFrame(data[self.feature[0]])
#
#         for i in range(1, len(self.feature)):
#             result = pd.concat([result, data[self.feature[i]]], axis=1)
#
#         # Return Dataset
#         return result
#
#
# # Description = It converts data according to each feature selection method
# #               If PCA is reset column name
# #               If RandomSelect is randomly determines features
# #               If CustomSelect is select specific features
# # Input  = Dataset, selected feature, number of feature
# # Output = (Processed) Dataset
# def makefeatureSubset(X, selection, n_feature):
#     selection.set_params(n_components=n_feature)
#     x_result = selection.fit_transform(X)
#     x_result = pd.DataFrame(x_result)
#
#     # Reset column name
#     if type(selection) == type(PCA()):
#         if n_feature == 3:
#             x_result.columns = ["Principle-1", "Principle-2", "Principle-3"]
#
#         elif n_feature == 4:
#             x_result.columns = ["Principle-1", "Principle-2", "Principle-3", "Principle-4"]
#
#         elif n_feature == 5:
#             x_result.columns = ["Principle-1", "Principle-2", "Principle-3", "Principle-4", "Principle-5"]
#
#     return x_result
#
#
# def AutoML(X, y=None, scale_col=None, encode_col=None, scalers=None, encoders=None,
#            features=None, feature_param=None, models=None, model_param=None,
#            scores=None, score_param=None):
#     # Set Encoder
#     if encoders is None:
#         # encode = [OrdinalEncoder(), OneHotEncoder(), LabelEncoder()]
#
#         # test용 인코더
#         encode = [OrdinalEncoder()]
#     else:
#         encode = encoders
#
#     # Set Scaler
#     if scalers is None:
#         # scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
#
#         # test용 스케일러
#         scale = [StandardScaler()]
#     else:
#         scale = scalers
#
#     # Set Feature
#     # features 입력이 None이라면 feature selection방법을 [모든 feature 선택, PCA, Selected_feature]로 설정
#     # selection 방법에 맞는 parameter도 기본값들로 설정
#     if features is None:
#         # feature = [PCA(), RandomSelect(), CustomSelect()]
#         feature = [CustomSelect()]
#         customSelectParameter = [["longitude", "latitude"], ["total_rooms", "total_bedrooms"],
#                                  ["longitude", "latitude", "total_rooms", "total_bedrooms"],
#                                  ["total_rooms", "total_bedrooms", "population", "households", "median_income"]]
#         feature_parameter = [customSelectParameter]
#
#     else:
#         feature = features
#         feature_parameter = feature_param
#
#     # Set Model
#     if models is None:
#         # model = [KMeans(), GaussianMixture(), clarans(), DBSCAN(), OPTICS()]
#
#         # test용 모델
#         model = [KMeans(), GaussianMixture(), DBSCAN()]
#     else:
#         model = models
#
#     # Set Model parameter
#     if model_param is None:
#         # KMeas Clustering
#         model_parameter = [{'n_clusters': [2, 3], 'init': ["k-means++", "random"],
#                             'n_init': [1, 10], 'max_iter': [100, 200]},
#                            # GMM(EM) Clustering
#                            {'n_components': [2, 3], 'max_iter': [100, 200],
#                             'covariance_type': ["spherical", "tied"],
#                             'n_init': [1, 10], 'tol': [1e-5, 1e-3]},
#
#                            # Clarans Clustering
#                            # {'number_clusters': [2, 3, 4, 5], 'numlocal_minima': [2, 3, 4],
#                            #  'max_neighbor': [2, 3, 4, 5]},
#
#                            # DBSCAN Clustering
#                            {'eps': [0.3, 0.4], 'min_samples': [2, 3],
#                             'metric': ["euclidean"], 'p': [1, 2, 3],
#                             'algorithm': ["auto", "ball_tree"]},
#
#                            # # Optics Clustering
#                            # {'eps': [0.3, 0.4, 0.5],
#                            #  'min_samples': [2, 3, 4, 5], 'p': [1, 2, 3],
#                            #  'cluster_method': ["xi", "dbscan"],
#                            #  'algorithm': ["auto", "ball_tree", "kd_tree", "brute"]}
#                            ]
#
#     else:
#         model_parameter = model_param
#
#     # Set Score
#     if scores is None:
#         score = []
#     else:
#         score = scores
#
#     # Set Score parameter
#     if score_param is None:
#         score_parameter = [None]
#     else:
#         score_parameter = score_param
#
#     cv = [(slice(None), slice(None))]
#
#     # 첫 15가지(feature selection 3, model 5)결과를 저장하는 score배열
#     # [  PCA , Model1][  PCA , Model2]...[  PCA , Model5]
#     # [Random, Model1][Random, Model2]...[Random, Model5]
#     # [Custom, Model1][Custom, Model2]...[Custom, Model5]
#     # firstScore = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
#     # firstScoreScaler = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
#     # firstScoreEncoder = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
#     # firstScoreFeature = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
#     # firstScoreModel = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
#     # firstScoreParameter = [[None, None, None, None, None], [None, None, None, None, None],
#     #                        [None, None, None, None, None]]
#     firstScore = [[0.36422508875626597, 0.3583197737781512, -0.18922996991959803, 0, 0],
#                   [0.5226162531098414, 0.5021989520509015, 0.15173322443353157, 0, 0],
#                   [0.6442539672913747, 0.5735714041022789, 0.7001271118588158, 0, 0]]
#     firstScoreScaler = [[StandardScaler(), StandardScaler(), StandardScaler(), None, None],
#                         [StandardScaler(), StandardScaler(), StandardScaler(), None, None],
#                         [StandardScaler(), StandardScaler(), StandardScaler(), None, None]]
#     firstScoreEncoder = [[OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), None, None],
#                          [OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), None, None],
#                          [OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), None, None]]
#     firstScoreFeature = [[['Principle-1', 'Principle-2', 'Principle-3'], ['Principle-1', 'Principle-2', 'Principle-3'],
#                           ['Principle-1', 'Principle-2', 'Principle-3'], None, None],
#                          [['housing_median_age', 'ocean_proximity'],
#                           ['longitude', 'median_income', 'ocean_proximity'],
#                           ['ocean_proximity', 'population'], None, None],
#                          [["longitude", "latitude"], ["longitude", "latitude"], ["total_rooms", "total_bedrooms"], None,
#                           None]]
#     firstScoreModel = [[KMeans(init='random', max_iter=100, n_clusters=3),
#                         GaussianMixture(covariance_type='spherical', n_components=2, n_init=10, tol=1e-05),
#                         DBSCAN(eps=0.4, min_samples=3, p=1), None, None],
#                        [KMeans(max_iter=100, n_clusters=2, n_init=1),
#                         GaussianMixture(covariance_type='spherical', n_components=2, tol=1e-05),
#                         DBSCAN(eps=0.3, min_samples=3, p=1), None, None],
#                        [KMeans(init='random', n_clusters=3, n_init=1),
#                         GaussianMixture(covariance_type='spherical', n_components=4), DBSCAN(min_samples=4, p=2), None,
#                         None]]
#     firstScoreParameter = [[{'init': 'random', 'max_iter': 100, 'n_clusters': 3, 'n_init': 10},
#                             {'covariance_type': 'spherical', 'max_iter': 100, 'n_components': 2, 'n_init': 10,
#                              'tol': 1e-05},
#                             {'algorithm': 'auto', 'eps': 0.4, 'metric': 'euclidean', 'min_samples': 3, 'p': 1}, None,
#                             None],
#                            [{'init': 'k-means++', 'max_iter': 100, 'n_clusters': 2, 'n_init': 1},
#                             {'covariance_type': 'spherical', 'max_iter': 100, 'n_components': 2, 'n_init': 1,
#                              'tol': 1e-05},
#                             {'algorithm': 'auto', 'eps': 0.3, 'metric': 'euclidean', 'min_samples': 3, 'p': 1}, None,
#                             None],
#                            [{'init': 'random', 'n_clusters': 3, 'n_init': 1},
#                             {'covariance_type': 'spherical', 'n_components': 4, 'n_init': 1, 'tol': 0.001},
#                             {'algorithm': 'auto', 'eps': 0.5, 'min_samples': 4, 'p': 2}, None, None]]
#
#     # purity를 통해 15가지를 -> 3가지로
#     secondScore = [0, 0, 0]
#     secondScoreScaler = [None, None, None]
#     secondScoreEncoder = [None, None, None]
#     secondScoreFeature = [None, None, None]
#     secondScoreModel = [None, None, None]
#     secondScoreParameter = [None, None, None]
#
#     ####################################################################
#     # Iterate
#     # for i in scale:
#     #     for j in encode:
#     #         # Scaling
#     #         df_scaled = pd.DataFrame(i.fit_transform(X[scale_col]))
#     #         df_scaled.columns = scale_col
#     #
#     #         # Encoding
#     #         if encode_col is not None:
#                 df_encoded = j.fit_transform(X[encode_col])
#                 df_encoded = pd.DataFrame(df_encoded)
#                 df_encoded.columns = encode_col
#                 df_prepro = pd.concat([df_scaled, df_encoded], axis=1)
#     #
#     #         else:
#     #             df_prepro = df_scaled
#     #
#     #         print(df_prepro)
#     #
#     #         # feature selection (find feature subset : PCA, random select, custom select)
#     #         featureIndex = 0
#     #         for z, z_param in zip(feature, feature_parameter):
#     #             modelIndex = 0
#     #             for m in model:
#     #                 for z_param_index in z_param:
#     #                     # feature selection 이 PCA라면 n_components 3,4,5에 대해 반복
#     #                     # feature selection 이 RandomSelect라면 n_components 3,4,5에 대해 반복
#     #                     # feature selection 이 CustomSelect라면 정해진 subset에 대해 반복
#     #                     # 해당 selection과 parameter에 맞는 feature subset이 나옴
#     #                     df_featureSubset = makefeatureSubset(df_prepro, z, z_param_index)
#     #
#     #                     gridSearch = GridSearchCV(estimator=m, param_grid=model_parameter[modelIndex],
#     #                                               scoring=cv_silhouette_scorer, cv=cv)
#     #                     # 그리드 서치.fit해줌
#     #                     result = gridSearch.fit(df_featureSubset)
#     #                     best_model = result.best_estimator_
#     #                     best_params = result.best_params_
#     #                     pred = best_model.fit_predict(df_featureSubset)
#     #                     score = silhouette_score(df_featureSubset, pred)
#     #                     print("현재 selection : ", z, "\n현재 모델 : ", m)
#     #                     print(best_model)
#     #                     print(best_params)
#     #                     print("score: ", score)
#     #
#     #                     if firstScore[featureIndex][modelIndex] == 0 or firstScore[featureIndex][modelIndex] < score:
#     #                         print(featureIndex)
#     #                         print(modelIndex)
#     #                         print(i)
#     #                         firstScore[featureIndex][modelIndex] = score
#     #                         firstScoreScaler[featureIndex][modelIndex] = i
#     #                         firstScoreEncoder[featureIndex][modelIndex] = j
#     #                         firstScoreFeature[featureIndex][modelIndex] = df_featureSubset.columns
#     #                         firstScoreModel[featureIndex][modelIndex] = best_model
#     #                         firstScoreParameter[featureIndex][modelIndex] = best_params
#     #
#     #                 modelIndex += 1
#     #             featureIndex += 1
#     #
#     # for i in range(0, 3):
#     #     for j in range(0, 5):
#     #         print("최종 결과", i, " ", j)
#     #         print(firstScoreScaler[i][j])
#     #         print(firstScoreEncoder[i][j])
#     #         print(firstScoreFeature[i][j])
#     #         print(firstScoreModel[i][j])
#     #         print(firstScoreParameter[i][j])
#     #         print(firstScore)
#     #         print(print())
#
#     # Purity를 구하는 for문
#     for a in range(1, 3):
#         for b in range(0, 3):
#             # scale_col을 scaling 해주기 => X[scale_col]
#             if firstScoreScaler[a][b] is not None:  # 해당하는 scaler가 있다면
#                 df_first_scaled = pd.DataFrame(firstScoreScaler[a][b].fit_transform(X[scale_col]))
#                 df_first_scaled.columns = scale_col
#             # else:   # 해당하는 scaler가 없다면
#             # df_first_scaled = X
#
#             # encode_col을 encoding 해주기 => X[encode_col]
#             if firstScoreEncoder[a][b] is not None:  # 해당하는 encoder가 있다면
#                 df_first_encoded = pd.DataFrame(firstScoreEncoder[a][b].fit_transform(X[encode_col]))
#                 df_first_encoded.columns = encode_col
#                 # scaled랑 합쳐준다
#                 df_score_and_encode = pd.concat([df_first_scaled, df_first_encoded], axis=1)
#             # else:   # 해당하는 encoder가 없다면
#             # df_score_and_encode = df_first_scaled
#
#             # print("**** Combination of Score and Encode ****\n")
#             # print(df_score_and_encode)
#
#             # scaling과 encoding한 dataframe에서 feature_selection에서 나온 feature들만 추출
#             if firstScoreFeature[a][b] is not None:
#                 first_fture = []
#                 for k in firstScoreFeature[a][b]:
#                     first_fture.append(k)
#
#                 df_new_score_and_encode = df_score_and_encode[first_fture]
#                 print("**** Apply feature selection ****\n")
#                 print(df_new_score_and_encode)
#
#             df_values = df_new_score_and_encode.values
#
#             # feature_selection한 결과물로 dataframe에서 추출했으면..?
#             # 이걸로 model을 fit_predict해준다!
#             if firstScoreModel[a][b] is not None:
#                 pred_val = firstScoreModel[a][b].fit_predict(df_new_score_and_encode)
#                 # print("predict value shape: {}".format(pred_val.shape))
#                 # print("**** Predicted Value ****\n")
#                 # print(pred_val)
#
#             min_y = np.min(y)
#             max_y = np.max(y)
#             gap = max_y - min_y
#             gap /= len(np.unique(pred_val))
#
#             labels = []
#             for i in range(len(np.unique(pred_val))):
#                 labels.append(i)
#
#             temp_df = pd.cut(y["median_house_value"], bins=len(np.unique(pred_val)), labels=labels, include_lowest=True)
#             temp_df = temp_df.to_numpy()
#
#             print("**** Purity Score ****")
#             purityScore=purity_score(temp_df, pred_val)
#             print(purityScore)
#
#             if purityScore > secondScore[a]:
#                 secondScore[a]=purityScore
#                 secondScoreScaler[a] = firstScoreScaler[a][b]
#                 secondScoreEncoder[a] = firstScoreEncoder[a][b]
#                 secondScoreFeature[a]= firstScoreFeature[a][b]
#                 secondScoreModel[a] = firstScoreModel[a][b]
#                 secondScoreParameter[a] = firstScoreParameter[a][b]
#
#     print(secondScore)
#     print(secondScoreScaler)
#     print(secondScoreEncoder)
#     print(secondScoreFeature)
#     print(secondScoreModel)
#     print(secondScoreParameter)
#
#     for i in range(1,3):
#         # scale_col을 scaling 해주기 => X[scale_col]
#         if secondScoreScaler[i] is not None:  # 해당하는 scaler가 있다면
#             df_second_scaled = pd.DataFrame(secondScoreScaler[i].fit_transform(X[scale_col]))
#             y_second_scaled=pd.DataFrame(secondScoreScaler[i].fit_transform(y))
#             y_second_scaled.columns=y.columns
#             df_second_scaled.columns = scale_col
#
#         # encode_col을 encoding 해주기 => X[encode_col]
#         if secondScoreEncoder[i] is not None:  # 해당하는 encoder가 있다면
#             df_second_encoded = pd.DataFrame(secondScoreEncoder[i].fit_transform(X[encode_col]))
#             df_second_encoded.columns = encode_col
#             # scaled랑 합쳐준다
#             df_score_and_encode = pd.concat([df_second_scaled, df_second_encoded], axis=1)
#
#         # scaling과 encoding한 dataframe에서 feature_selection에서 나온 feature들만 추출
#         if secondScoreFeature[i] is not None:
#             second_fture = []
#             for k in secondScoreFeature[i]:
#                 second_fture.append(k)
#             df_new_score_and_encode = df_score_and_encode[second_fture]
#             df_new_score_and_encode_y=pd.concat([df_new_score_and_encode,y_second_scaled],axis=1)
#
#         model = secondScoreModel[i]
#         print(secondScoreScaler[i])
#         print(secondScoreEncoder[i])
#         print(secondScoreFeature[i])
#         print(secondScoreModel[i])
#         print(secondScoreParameter[i])
#         cluster_no=secondScoreModel[i].fit(df_new_score_and_encode)
#         label=cluster_no.labels_
#         print(label)
#
#         fig = px.scatter(df_new_score_and_encode, color=label)
#         fig.show()
#
#         pred_no = cluster_no.fit_predict(df_new_score_and_encode)
#         score = silhouette_score(df_new_score_and_encode, pred_no)
#         print("Silhouette score = ", score)
#
#         cluster_yes = secondScoreModel[i].fit(df_new_score_and_encode_y)
#         label=cluster_yes.labels_
#         print(label)
#
#         fig = px.scatter(df_new_score_and_encode_y, color=label)
#         fig.show()
#
#         pred_yes = cluster_yes.fit_predict(df_new_score_and_encode_y)
#         score = silhouette_score(df_new_score_and_encode_y, pred_yes)
#         print("Silhouette score = ", score)
#
#
#
# ##############################################################################
# # Dataset = California housing price
# # Feature = longitude, latitude, housing_median_age, total_rooms, total_bedrooms
# #           population, households, median_income, ocean_proximity
# # Target = median_house_value
#
# # Number of dataset = 20640
# # Numerical value = longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
# #                   population, households, median_income, median_house_value
# # Categorical value = ocean_proximity
#
# df = pd.read_csv("housing.csv")
#
# feature_label = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
#                  'population', 'households', 'median_income', 'ocean_proximity']
# target_label = ['median_house_value']
#
# # Print housing data's information
# # print("\n***************** housing ****************")
# # print(df.head())
# #
# # print("\n************** Description ***************")
# # print(df.describe())
# #
# # print("\n************** Information ***************")
# # print(df.info())
#
# # Check null value
# # print("\n************** Check null ***************")
# # print(df.isna().sum())
#
# # Fill null value
# df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)
# # print("\n******** Check null (Cleaned data) ******")
# # print(df.isna().sum())
#
# # Remove Outliers with z-score
# # Description = Use the z-score to handle outlier over mean +- 3SD
# # Input  = dataframe's column
# # Output = index
# df_cate = df['ocean_proximity']
#
# def find_outliers(col):
#     z = np.abs(stats.zscore(col))
#     idx_outliers = np.where(z > 3, True, False)
#     return pd.Series(idx_outliers, index=col.index)
#
#
# for n in range(len(feature_label)):
#     idx = None
#     idx = find_outliers(df.iloc[:, n])
#     df = df.loc[idx == False]
#
# # print("\n******** Removed Outlier ******")
# # print(df.info())
#
# # Set X, y data
# y_data = df.loc[:, target_label]
# X_data = df.drop(target_label, axis=1)
#
# scale_col = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
#              "median_income"]
# end_col = ["ocean_proximity"]
#
# AutoML(X_data,y_data, scale_col=scale_col, encode_col=end_col, models=None, model_param=None)
