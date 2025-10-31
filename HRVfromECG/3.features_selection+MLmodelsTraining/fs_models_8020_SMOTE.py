# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:08:51 2023

@author: umbym
"""

#%% IMPORT
import numpy as np
import pandas as pd
import joblib
# from imblearn.over_sampling import SMOTE
import copy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from mrmr import mrmr_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#%% 8020 Features Scores
def min_max_scaling(arr):
    min_val = min(arr)
    max_val = max(arr)
    scaled_arr = [(x - min_val) / (max_val - min_val) for x in arr]
    return np.array(scaled_arr)

def select_features(X_train, y_train, best_params): 
 # configure to select a subset of features
 if 'anova__k' in best_params:
     fs = SelectKBest(score_func=f_classif, k=best_params['anova__k'])
     fs.fit(X_train, y_train)
     score = min_max_scaling(fs.scores_)
    #  score = minmaxScaler.fit_transform([fs.scores_])
     # rank = fs.scores_.argsort()
 elif 'mutual_info__k' in best_params:
     fs = SelectKBest(score_func=mutual_info_classif, k=best_params['mutual_info__k'])
     fs.fit(X_train, y_train)
     score = min_max_scaling(fs.scores_)
     # rank = fs.scores_.argsort()
 elif 'PCA__n_components' in best_params:
     fs = PCA(n_components=best_params['PCA__n_components'])
     fs.fit(X_train, y_train)
     score = min_max_scaling(fs.mean_)
     # rank = fs.mean_.argsort()
 return score# fit the model

from sklearn.metrics import confusion_matrix
def confmatrix_metrics(y_test,y_pred):
    VN, FP, FN, VP = map(int, confusion_matrix(y_test, y_pred).ravel())
    conf_mat =[[VP,FN],[FP,VN]] 
    accuracy = (VP+VN)/(VP+VN+FN+FP) if (VP+VN+FN+FP) != 0 else 0
    recall = VP/(VP+FN) if (VP+FN) != 0 else 0
    specificity = VN/(FP+VN) if (FP+VN) != 0 else 0
    precision = VP/(VP+FP) if (VP+FP) != 0 else 0
    F1 = (2*recall*precision/(recall+precision)) if (recall+precision) != 0 else 0
    return accuracy, precision, specificity, recall, F1, conf_mat

#%% Models and feature selection definition

# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

# Define models
classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVC': SVC(),
    'NB': GaussianNB(),
    # 'DT': DecisionTreeClassifier(),
    'BAG': BaggingClassifier(),
    'XGB': XGBClassifier(),#
    'RF': RandomForestClassifier()
}

# Define feature selection methods
feature_selectors = {
    'anova': SelectKBest(score_func=f_classif),
    'mutual_info': SelectKBest(score_func=mutual_info_classif),
    'PCA': PCA()
}

# Create pipelines
pipelines = []
for fs_name, fs_method in feature_selectors.items():
    for clf_name, clf in classifiers.items():
        pipeline = Pipeline(steps=[(fs_name, fs_method), (clf_name, clf)])
        pipelines.append(Pipeline(steps=[(fs_name, fs_method), (clf_name, clf)]))

#%% READ DATA
sig = ["HR","HRO"]
feat = ["1235","all"]
paz = ["CAP","TO","CAP+TO","CAP+DREAMS","CAP+ISRUC","CAP+TO+DREAMS","CAP+TO+ISRUC","CAP+DREAMS+ISRUC","CAP+TO+DREAMS+ISRUC"]# 
model_name_list = ["%s_%s_%s" % (p,f,s) for p in paz for f in feat for s in sig]
dataset_names_TR = ["%s_TR_%s_%s" % (p,f,s) for p in paz for f in feat for s in sig]
dataset_names_TE = ["%s_TE_%s_%s" % (p,f,s) for p in paz for f in feat for s in sig]
# model_name_list_for = [model_name_list[3],model_name_list[5]]
# dataset_names_TR_for = [dataset_names_TR[3],dataset_names_TR[5]]
# dataset_names_TE_for = [dataset_names_TE[3],dataset_names_TE[5]]


for ds_TR_name, ds_TE_name, model_name in zip(dataset_names_TR, dataset_names_TE, model_name_list):
    # define dataset
    # df = pd.read_csv("%s_NOcorr_featTOT.csv" % (ds_name))
    # dataset = df.values

    df = pd.read_csv("feat_eval/%s_NOcorr_RnR.csv" % (ds_TR_name))
    dataset_TR = df.values
    df = pd.read_csv("feat_eval/%s_NOcorr_RnR.csv" % (ds_TE_name))
    dataset_TE = df.values

    # split into input (X) and output (y) variables
    # X = dataset[:, :-1]
    # y = dataset[:,-1]
    X_train = dataset_TR[:, 1:-1].astype(np.float64)
    y_train = dataset_TR[:,-1].astype(np.float64)
    X_test = dataset_TE[:, 1:-1].astype(np.float64)
    y_test = dataset_TE[:,-1].astype(np.float64)

    IDs_test = dataset_TE[:, 0]

    # oversample with SMOTE
    # oversample = SMOTE(sampling_strategy=100/105, n_jobs=-1)
    # X_train, y_train = oversample.fit_resample(X, y)
    # X_train, X_VAL, y_train, y_VAL = train_test_split(X, y, test_size = 0.2)

    #%% GRIDS
    # Preprocessing parameters mapped in a single dictionary
    preprocessing_params = {
        'anova': {'anova__k': list(range(1, 11))},
        'mutual_info': {'mutual_info__k': list(range(1, 11))},
        'PCA': {'PCA__n_components': list(range(1, 11))}
    }

    # Model-specific hyperparameters
    model_params = {
        'KNN': {
            'n_neighbors': list(range(1,int(np.fix(0.5 * X_train.shape[0])),2)),
            'p': [1, 2],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'SVC': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'class_weight': [None, 'balanced']
        },
        'NB': {'var_smoothing': np.logspace(0, -9, num=100)},
        # 'DT': {
        #     'criterion': ['gini', 'entropy'],
        #     'max_depth': list(range(1, 21, 2)),
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'min_impurity_decrease': [0.0, 0.01, 0.1]
        # },
        'BAG': {
            'n_estimators': [10, 30, 70, 100, 300, 700, 1000],
            'max_samples': [0.5, 0.7, 1.0],
            'max_features': [0.5, 0.7, 1.0]
        },
        'XGB': {
            'max_depth': list(range(2, 13, 2)),
            'n_estimators': [50, 100, 200, 300, 500, 700],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
            'min_child_weight': [1, 3, 5, 10],
            'gamma': [0, 0.1, 0.2]
        },
        'RF': {
            'criterion': ['gini', 'entropy'],
            'max_depth': 	[None, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }
    }

    # Generate the grid in a single function
    def generate_grids(preprocessing_methods, model_params):
        grids = [
            {**preprocessing_params[preproc], **{f'{model}__{k}': v for k, v in params.items()}}
            for preproc in preprocessing_methods
            for model, params in model_params.items()
        ]
        return grids

    # Call the function with preprocessing methods and model parameters
    grids = generate_grids(preprocessing_params.keys(), model_params)
    
    # pipelines_for = pipelines[0:5]
    # grids_for = grids[9:12]

    #%% TRAIN
    excel1_list = []
    excel2_list = []
    excel2_list.append(y_test)
    excel3_list = np.zeros((X_train.shape[1],len(preprocessing_params)+1))
    excel1_row = []
    excel3_col = []
    labels = df.columns.values[1:-1]
    score_overall = np.zeros((X_train.shape[1],), dtype=int)
    col = 0
    for idx, (pipe, grid) in enumerate(zip(pipelines, grids)):    
        model_8020 = GridSearchCV(pipe, grid, scoring='f1', n_jobs=-1, cv=cv, verbose=2, error_score="raise")
        model_8020.fit(X_train, y_train)

        best_model = model_8020.best_estimator_

        name  = "%s+%s" % (model_8020.best_estimator_.steps[0][0], model_8020.best_estimator_.steps[1][0])        
        filename = f"Models/{model_name}/{model_name}_model_{name}.pkl"
        joblib.dump(best_model, filename)

        if idx % len(model_params) == 0:
            score = select_features(X_train, y_train, model_8020.best_params_)
            for i in range(len(score)):
                excel3_list[i,col] = score[i]
            excel3_col.append(model_8020.best_estimator_.steps[0][0])
            score_overall = score_overall + score
            col = col+1
        
        y_pred = best_model.predict(X_test)
        accuracy_TE, precision_TE, specificity_TE, recall_TE, F1_TE, conf_mat_TE = confmatrix_metrics(y_test, y_pred)
        # y_pred_VAL = best_model.predict(X_VAL)
        # accuracy_VAL, precision_VAL, specificity_VAL, recall_VAL, F1_VAL, conf_mat_VAL = confmatrix_metrics(y_VAL, y_pred_VAL)        
        
        excel1_list.append([model_8020.best_params_, model_8020.best_score_, accuracy_TE, precision_TE, specificity_TE, recall_TE, F1_TE, conf_mat_TE])#accuracy_TR, precision_TR, specificity_TR, recall_TR, F1_TR, conf_mat_TR,
        excel1_row.append(name)
        excel2_list.append(y_pred)
        

    X = pd.DataFrame(X_train)
    y = pd.Series(y_train)   
    flag = 1
    for pipe, grid in zip(pipelines[0:len(model_params)],grids[0:len(model_params)]):
        sel_feat_list = []
        model_scores = []
        best_models = []
        p = copy.deepcopy(pipe)
        g = copy.deepcopy(grid)
        del p.steps[0]
        del g[next(iter(g))]
        for k in list(range(1,11)):
            sel_feat = mrmr_classif(X=X, y=y, K=k)
            sel_feat_list.append(sel_feat)
            X_train_fs = X_train[:,sel_feat]
            # X_test_fs = X_test[:,sel_feat]             

            model_8020 = GridSearchCV(p, g, scoring='f1', n_jobs=-1, cv=cv, verbose=2)
            model_8020.fit(X_train_fs, y_train)
            best_models.append(model_8020.best_estimator_)
            model_scores.append(model_8020.best_score_)
            if model_8020.best_score_ == 1:
                break
        idx = model_scores.index(max(model_scores))
        best_model = best_models[idx]
        sel_feat = sel_feat_list[idx]

        y_pred = best_model.predict(X_test[:,sel_feat])        
        accuracy_TE, precision_TE, specificity_TE, recall_TE, F1_TE, conf_mat_TE = confmatrix_metrics(y_test, y_pred)       
        # y_pred_VAL = best_model.predict(X_VAL[:,sel_feat])
        # accuracy_VAL, precision_VAL, specificity_VAL, recall_VAL, F1_VAL, conf_mat_VAL = confmatrix_metrics(y_VAL, y_pred_VAL)

        name  = "mrmr+%s" % (model_8020.best_estimator_.steps[0][0])
        filename = f"Models/{model_name}/{model_name}_model_{name}.pkl"
        joblib.dump(best_model, filename)

        if flag == 1:
            score = mrmr_classif(X=X, y=y, K=X_train.shape[1]+1)
            feat_pos = [None]*(X_train.shape[1])
            for pos,idx in enumerate(score):
                feat_pos[idx] = pos+1
            feat_pos = np.array(feat_pos)
            for i in range(len(feat_pos)):
                excel3_list[i,col] = feat_pos[i]
            excel3_col.append("mrmr")
            flag = 0

        model_8020.best_params_["mrmr__K"] = len(sel_feat)
        excel1_list.append([model_8020.best_params_, model_8020.best_score_, accuracy_TE, precision_TE, specificity_TE, recall_TE, F1_TE, conf_mat_TE]) #accuracy_TR, precision_TR, specificity_TR, recall_TR, F1_TR, conf_mat_TR,
        excel1_row.append(name)
        excel2_list.append(y_pred)

    score_overall = score_overall#[0,:]  
    ind = np.argsort(-score_overall)
    top_scores_overall = score_overall[ind]
    top_labels_overall = labels[ind]
    
    # %% Write Excel
    excel1 = pd.DataFrame(excel1_list, index=excel1_row, columns=['best_config', 'model_best_score', 'acc_Test', 'prec_Test', 'spec_Test', 'recall_Test', 'F1_Test', 'conf_mat_Test'])# 'acc_TR', 'prec_TR', 'spec_TR', 'recall_TR', 'F1_TR', 'conf_mat_TR',
    excel2 = pd.DataFrame(excel2_list, index=['actual_class'] + excel1_row, columns=IDs_test)
    excel3 = pd.DataFrame(excel3_list, index=labels.tolist(), columns=excel3_col)
    excel4 = pd.DataFrame(np.stack((top_labels_overall, top_scores_overall), axis=1).tolist(), index=list(range(1,X_train.shape[1]+1)), columns=['top_feat', 'top_scores'])

    
    with pd.ExcelWriter('Results/%s_8020.xlsx' % (model_name)) as writer:
        excel1.to_excel(writer, sheet_name='model_performances')
        excel2.to_excel(writer, sheet_name='predictions')
        excel3.to_excel(writer, sheet_name='features')
        excel4.to_excel(writer, sheet_name='overall_features')