"""
Created on Wed Oct 12 18:33:50 2022

@author: Piero Rendina
"""

import pandas as pd
import numpy as np
import toolbox_02450
from sklearn import tree
import seaborn as sns
from toolbox_02450 import windows_graphviz_call, mcnemar
from os import getcwd
import sklearn
import matplotlib.pyplot as plt
import plot_style
from matplotlib.image import imread
#from feature_selection import *


def plot_confusion_matrix(y_true, y_pred, model):
    plt.figure()
    true_0  = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    false_0 = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    false_1 = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    true_1  = np.sum(np.logical_and(y_true == 1, y_pred == 1))

    accuracy = (true_0 + true_1)/(true_0+true_1+false_0+false_1)
    recall = true_1/(true_1+false_0)
    precision = true_1/(true_1+false_1)

    print('true  %s: %d' % ('survived', true_0))
    print('false %s: %d' % ('survived', false_0))
    print('true  %s: %d' % ('dead', true_1))
    print('false %s: %d' % ('dead', false_1))
    print('accuracy = %1.2f %%' % (accuracy * 100))
    print('precision = %1.2f %%' % (precision * 100))
    print('recall = %1.2f %%' % (recall * 100))

    cf_matrix = np.array([[true_0, false_1],[false_0, true_1]])
    values = cf_matrix/np.sum(cf_matrix)
    group_names = ['TN','FP','FN','TP']
    group_counts = ["{0:}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix '+model, size=22);
    ax.set_xlabel('Predicted Values',size=18)
    ax.set_ylabel('Actual Values',size=18);
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'],size=16)
    ax.yaxis.set_ticklabels(['False','True'],size=16)

    ## Display the visualization of the Confusion Matrix.
    plt.show()




# %%
# Load the original dataset
data_path = "heart_failure_clinical_records_dataset.csv"
data_df = pd.read_csv(data_path, delimiter=',')

columns = data_df.columns


# the feature selection will provide the best subset of features to work with
# classification_features = best_features
debug = False
if not debug:
    classification_features = columns.delete(-1)
else:
    #classification_features = ['serum_creatinine','ejection_fraction']
    classification_features = best_features
classification_df = data_df[classification_features].copy()

for feature in classification_features:
    classification_df[feature] = (classification_df[feature]-classification_df[feature].mean())/classification_df[feature].std()

X = np.asarray(classification_df)
y = np.array(data_df['DEATH_EVENT'].copy())
# Shuffle the data set to avoid introducing bias
X, y = sklearn.utils.shuffle(X, y, random_state = 1)


# %% Two-levels cross-validation for the LogisticRegression model
# Setting the hyperparameters to perform the two-level cross-validation 
K_outer = 10
K_inner = 10
lambdas = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 5, 6, 7, 10, 12, 14])
n_models = len(lambdas)
# logistic regressor requires the inverse of lambda as constructor parameter
C = 1/lambdas
# first level cross-validation
CV_outer = sklearn.model_selection.KFold(n_splits=K_outer,shuffle=True, random_state=0)
# array to store the accuracy computed on the outer set
outer_results = np.zeros(K_outer)
logistic_results = []


for outer_idx, (train_idx, test_idx) in enumerate(CV_outer.split(X)):
    X_train, X_test = X[train_idx, :], X[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]           
    # second level cross-validation
    CV_inner = sklearn.model_selection.KFold(n_splits=K_inner,shuffle=True, random_state=1)
    inner_results = np.zeros((n_models, K_inner))
    for inner_fold_idx, (train_inner_idx, test_inner_idx) in enumerate(CV_inner.split(X_train)):
        X_train_inner, X_test_inner = X_train[train_inner_idx, :], X_train[test_inner_idx, :]
        y_train_inner, y_test_inner = y_train[train_inner_idx], y_train[test_inner_idx]  
        # train the different models on the same split
        for idx, c in enumerate(C):
            logistic_model = sklearn.linear_model.LogisticRegression(penalty="l2", C=c).fit(X_train_inner, y_train_inner)
            y_pred_inner = logistic_model.predict(X_test_inner)
            err_inner = np.sum((y_pred_inner != y_test_inner))/len(y_pred_inner)
            inner_results[idx, inner_fold_idx] = err_inner
        
    # compute the generalization error for each model using the inner folds
    performances = np.zeros(n_models)
    for model_idx in range(n_models):
        performances[model_idx] = np.sum((len(X_test_inner)/len(X_train))*inner_results[model_idx])
    
    best_idx = np.argmin(performances, axis=0)
    best_model = sklearn.linear_model.LogisticRegression(penalty="l2", C=C[best_idx]).fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    best_error = np.sum(y_pred != y_test)/len(y_test)
    best_dict = {"lambda": lambdas[best_idx], "logistic_error": best_error}
    outer_results[outer_idx] = best_error
    logistic_results.append(best_dict)
    
final_error = np.sum(len(X_test)/len(X)*outer_results)
# Create a table with the results of the two-levels cross-validation
logistics_df = pd.DataFrame.from_records(logistic_results)
print(logistics_df)
print(f"Estimated logistic regression error = {final_error}")

# %% Two-levels cross-validation for the Baseline model
from scipy import stats
CV_outer = sklearn.model_selection.KFold(n_splits=K_outer,shuffle=True, random_state=0)
baseline_error = np.zeros(K_outer)
fold_idx = 0
for train_idx, test_idx in CV_outer.split(X):
    X_train, X_test = X[train_idx,:], X[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]
    y_pred = stats.mode(y_train, axis=0)[0]
    baseline_error[fold_idx] = np.sum(y_pred != y_test)/len(y_test)
    fold_idx += 1

final_baseline_error = np.sum((len(X_test)/len(X))*baseline_error)
baseline_df = pd.DataFrame(baseline_error, columns=['baseline_error'])
print(baseline_df)
print(f"Estimated baseline error = {final_baseline_error}")

# %% Two-levels cross-validation for the classification trees
min_samples_split_values = [3, 4, 5, 7, 10, 15, 20, 30, 34, 38, 40, 45, 50]
criterion = 'gini'
tree_results = []
n_models = len(min_samples_split_values)


for outer_idx, (train_idx, test_idx) in enumerate(CV_outer.split(X)):
    X_train, X_test = X[train_idx, :], X[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]           
    # second level cross-validation
    CV_inner = sklearn.model_selection.KFold(n_splits=K_inner,shuffle=True, random_state=1)
    inner_fold_idx = 0
    inner_results = np.zeros((n_models, K_inner))
    for train_inner_idx, test_inner_idx in CV_inner.split(X_train):
        X_train_inner, X_test_inner = X_train[train_inner_idx, :], X_train[test_inner_idx, :]
        y_train_inner, y_test_inner = y_train[train_inner_idx], y_train[test_inner_idx]  
        # train the different models on the same split
        for idx, min_samples_split in enumerate(min_samples_split_values):
            decision_tree = sklearn.tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split).fit(X_train_inner,y_train_inner)
            y_pred_inner = decision_tree.predict(X_test_inner)
            err_inner = np.sum((y_pred_inner != y_test_inner))/len(y_pred_inner)
            inner_results[idx, inner_fold_idx] = err_inner
        inner_fold_idx += 1
        
    # compute the generalization error for each model using the inner folds
    performances = np.zeros(n_models)
    for model_idx in range(n_models):
        performances[model_idx] = np.sum((len(X_test_inner)/len(X_train))*inner_results[model_idx])
    
    best_idx = np.argmin(performances, axis=0)
    best_model = sklearn.tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split_values[best_idx]).fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    best_error = np.sum(y_pred != y_test)/len(y_test)
    best_dict = {"min_samples_split_values": min_samples_split_values[best_idx], "decision_tree_error": best_error}
    outer_results[outer_idx] = best_error
    tree_results.append(best_dict)

final_error = np.sum(len(X_test)/len(X)*outer_results)
tree_df = pd.DataFrame.from_records(tree_results)
print(tree_df)
print(f"Estimated decision tree error = {final_error}")


#%% Comparison of the models 
from collections import Counter
logistic_models = np.zeros(K_outer)
tree_models = np.zeros(K_outer)
for idx, logistic_result,tree_result in zip(range(len(logistic_results)),logistic_results,tree_results):
    logistic_models[idx] = logistic_result.get("lambda")
    tree_models[idx] = tree_result.get("min_samples_split_values")
    
lambda_count = Counter(logistic_models)
min_samples_count = Counter(tree_models)
best_lambda = lambda_count.most_common(1)[0][0]
best_min_samples = int(min_samples_count.most_common(1)[0][0])

print("Best lambda = {0}, Best minimum number of samples = {1}".format(best_lambda, best_min_samples))

# %% Comparison of the three classification models
CV = sklearn.model_selection.LeaveOneOut()
yhat = []
y_true = []
y_est_proba = []
for i,(train_index, test_index) in enumerate(CV.split(X, y)):
    #print('Crossvalidation fold: {0}/{1}'.format(i+1,X.shape[0]))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    dy = []
    for model_idx in range(3):
        # Logistic regression case
        if model_idx == 0:
            logistic_regressor = sklearn.linear_model.LogisticRegression(C=1/best_lambda).fit(X_train, y_train)
            y_est = logistic_regressor.predict(X_test)
            y_est_proba.append(logistic_regressor.predict_proba(X_test)[:,1])
        # Baseline case
        if model_idx == 1:
            y_est = stats.mode(y_train, axis=0)[0]
        # Classification tree case
        if model_idx == 2:
            clt = sklearn.tree.DecisionTreeClassifier(min_samples_split=best_min_samples).fit(X_train, y_train)
            y_est = clt.predict(X_test)

        dy.append(y_est)

    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    i+=1

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true) 
y_est_proba = np.array(y_est_proba)
yhat[:,0] # predictions made by first classifier.

# Compute accuracy here.
accuracies = np.array([np.sum(yhat[:,i] == y_true)/len(yhat[:,i]) for i in range(3)])
toolbox_02450.rocplot(y_est_proba,y_true)
# %% Plot the confusion matrix for the three classifiers
model_names = ['logistic regression', 'baseline', 'classification tree']
for i,model_name in zip(range(3),model_names):
    plt.figure()
    plot_confusion_matrix(y_true, y_pred=yhat[:,i], model=model_name)
print("Accuracies of the classifiers: "+str(accuracies))


# %%  Compute the Jeffreys interval - Comparison of the models
alpha = 0.05
for first_model, second_model in [[0,1],[2,1],[0,2]]:
    [thetahat, CI, p] = mcnemar(y_true, yhat[:,first_model], yhat[:,second_model], alpha=alpha)
    print("theta = theta_A-theta_B point estimate", thetahat, " CI:", CI, " p-value", p)


# %% Train and test a final Logistic regression model
X_final_train, X_final_test = X[range(250),:],X[range(250,299,1),:]
y_final_train, y_final_test = y[range(250)], y[range(250,299,1)]
final_logistic_model = sklearn.linear_model.LogisticRegression(C=1/best_lambda).fit(X_final_train, y_final_train)
coeffs_df = pd.DataFrame(final_logistic_model.coef_, columns = classification_features)
print(coeffs_df)

sns.set_theme()
plt.grid()
toolbox_02450.rocplot(y_est_proba,y_true)

# %% Store and plot the trend of the logistic coefficients
coeffs_df = pd.DataFrame(columns = classification_features)
for lambda_ in np.linspace(0.0001, 50, num=100):
    logistic_model = sklearn.linear_model.LogisticRegression(C=1/lambda_).fit(X_final_train, y_final_train)
    coeffs_df.loc[len(coeffs_df)] = logistic_model.coef_.flatten()
    

plt.figure()
for column in classification_features:
    plt.plot(coeffs_df[column], label=column)
plt.title("Trend of the logistic regression coefficients")
plt.legend()



# %%
merged = tree_df.merge(logistics_df,left_index=True,right_index=True).merge(baseline_df,left_index=True,right_index=True)
print(merged)

# %% Training the final logistic regression
from sklearn.decomposition import IncrementalPCA

pca = IncrementalPCA(n_components = 2)
X_train_pca = pca.fit_transform(X_final_train)
X_test_pca = pca.transform(X_final_test)
logistic_pca = sklearn.linear_model.LogisticRegression(C=0.2).fit(X_train_pca, y_final_train)

y_pred_pca = logistic_pca.predict(X_test_pca)
error_pca = np.sum(y_final_test != y_pred_pca)/len(y_pred_pca)

# %% Plot the results on the decomposition of the dataset on the 
plt.figure(figsize = (20, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[y_final_train==0,0], X_train_pca[y_final_train==0,1])
plt.scatter(X_train_pca[y_final_train==1,0], X_train_pca[y_final_train==1,1])
plt.xlabel('Training 1st Principal Component')
plt.ylabel('Training 2nd Principal Component')
plt.title('Training Set Scatter Plot with labels indicated by colors i.e., (0) -> Violet and (1) -> Yellow')
plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[y_final_test==0,0], X_test_pca[y_final_test==0,1])
plt.scatter(X_test_pca[y_final_test==1,0], X_test_pca[y_final_test==1,1])
plt.xlabel('Test 1st Principal Component')
plt.ylabel('Test 2nd Principal Component')
plt.title('Test Set Scatter Plot with labels indicated by colors i.e., (0) -> Violet and (1) -> Yellow')
plt.show()

# %% Second plot
# Retrieve the model parameters.
b = logistic_pca.intercept_[0]
w1, w2 = logistic_pca.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

plt.figure(figsize = (24,10))
plt.subplot(1, 2, 1)

# Plot the data and the classification with the decision boundary.
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xd = np.array([x_min, x_max])
yd = m*xd + c

# Training plot
plt.subplot(1,2,1)
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, y_min, color='tab:orange', alpha=0.2)
plt.fill_between(xd, yd, y_max, color='tab:blue', alpha=0.2)
plt.scatter(X_train_pca[y_final_train==0,0], X_train_pca[y_final_train==0,1],s = 30, alpha = 0.5, label='survival')
plt.scatter(X_train_pca[y_final_train==1,0], X_train_pca[y_final_train==1,1],s = 30, alpha = 0.5, label='death')
plt.xlabel('Training 1st Principal Component')
plt.ylabel('Training 2nd Principal Component')
plt.legend(prop={'size':22})
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Scatter Plot for the Training Set', fontsize=26)
plt.xlabel('1st Principal Component', fontsize=20)
plt.ylabel('2nd Principal Component', fontsize=20)

# Test plot
plt.subplot(1,2,2)
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, y_min, color='tab:orange', alpha=0.2)
plt.fill_between(xd, yd, y_max, color='tab:blue', alpha=0.2)
plt.scatter(*X_test_pca[y_final_test==0].T, s = 30, alpha = 0.5, label='survival')
plt.scatter(*X_test_pca[y_final_test==1].T, s = 30, alpha = 0.5, label='death')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(prop={'size':22})
plt.title('Scatter Plot for the Test Set', fontsize=26)
plt.xlabel('1st Principal Component', fontsize=20)
plt.ylabel('2nd Principal Component', fontsize=20)
plt.show()


# %% Plot only the ejection fraction and the serum creatinine


model = sklearn.linear_model.LogisticRegression(C = 1.0/best_lambda).fit(X[:,[7,4]],y)
b = model.intercept_[0]
w1, w2 = model.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

x_min, x_max = X[:, 7].min() - 1, X[:, 7].max() + 1
y_min, y_max = X[:, 4].min() - 1, X[:, 4].max() + 1
xd = np.array([x_min, x_max])
yd = m*xd + c

# Training plot
plt.figure(figsize = (12, 8))
plt.scatter(X[y==0,7], X[y==0,4], label='surival')
plt.scatter(X[y==1,7], X[y==1,4], label='death')
plt.legend(prop={'size':18})
plt.title('Ejection fraction - Serum Creatinine scatter plot', fontsize=30)
plt.ylabel('Ejection fraction',fontsize=22)
plt.xlabel('Serum creatinine',fontsize=22)
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.fill_between(xd, yd, y_min, color='tab:orange', alpha=0.2)
plt.fill_between(xd, yd, y_max, color='tab:blue', alpha=0.2)
plt.xlabel('Serum Creatinine',fontsize=22)
plt.ylabel('Ejection Fraction',fontsize=22)
plt.title('Logistic regression results',fontsize=30)
