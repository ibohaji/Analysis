import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
import seaborn as sns
from plot_style import set_plot_theme

#!adjust path accordingly!
data_path = ''
data = pd.read_csv(data_path + 'heart_failure_clinical_records_dataset.csv', delimiter = ',')

#Setting the default theme for the plots
set_plot_theme()
pal = {
       'Train Error': '#9463A6',
       'Test Error': '#89A643'
       }
#change the colnames and value names for better clarity
data = data.set_axis(
    ["Age", "Anaemia", "CPK", "Diabetes", "EF", "hBP", "Platelets", "SC", "sNa", "Sex", "Smoking", "Time", "Death Event"],
    axis = 1
    )
measures = ["years","1000 mcg/L","Percentage","1000 kiloplatelets/mL","mg/dL","mEq/L", "Days"]

#normalize data and extract all features deemed relevant for regression
data_norm = data.copy()
for col in data_norm.columns:
    if sorted(list(data_norm[col].unique())) != [0, 1]:
        data_norm[col] = (data_norm[col] - data_norm[col].mean())/data_norm[col].std()
data_reg = data_norm[['SC', 'sNa', 'EF', 'Age', 'Time']].copy()
rel_predictors = data_norm.drop(['SC', 'Death Event'], axis = 1)

#%%
#Basic Model Setup

#create different subsets to compare different combinations of parameters
reg1 = data_reg[['SC', 'sNa']]
reg2 = data_reg[['SC', 'EF']]
reg3 = data_reg[['SC', 'sNa', 'Age']]
reg4 = data_reg[['SC', 'sNa', 'Time']]
reg5 = data_reg[['SC', 'EF', 'Age']]
reg6 = data_reg[['SC', 'EF', 'Time']]
reg7 = data_reg[['SC', 'sNa', 'EF', 'Age']]
reg_l = [reg1, reg2, reg3, reg4, reg5, reg6, reg7, 'forward', 'backward']

cvf = 10

#test the different feature combinations and store their errors
m_errs = np.empty((9, 4), dtype = object)
for ind, dat in enumerate(reg_l):
    cv = model_selection.KFold(n_splits = cvf, shuffle = True)
    errs = np.zeros((cvf, 2))

    if type(dat) != str:
        y = dat.values[:, 0]
        X = dat.values[:, 1:]
    else:
        #sequential feature selection
        y = data_norm[['SC']].values
        X = rel_predictors.values

        regressor = lm.LinearRegression()
        sfs = SequentialFeatureSelector(regressor, n_features_to_select = 2, direction = dat, cv = cvf)
        res = sfs.fit(X, y)
        selected_feats = rel_predictors.iloc[:, res.support_]
        X = selected_feats.values

    i = 0
    for train_i, test_i in cv.split(X):
        X_train, y_train = X[train_i, :], y[train_i]
        X_test, y_test = X[test_i, :], y[test_i]

        regressor = lm.LinearRegression().fit(X_train, y_train)
        y_pred1 = regressor.predict(X_train)
        y_pred2 = regressor.predict(X_test)

        err_train = np.square(y_train - y_pred1).sum()/len(y_train)
        err_test = np.square(y_test - y_pred2).sum()/len(y_test)
        errs[i, :] = err_train, err_test
        i += 1

    av_err_train, av_err_test = errs[:, 0].mean(), errs[:, 1].mean()

    if type(dat) != str:
        feats = dat.drop('SC', axis = 1).columns
        feats = [feat for feat in feats]
        feats = ' \n + '.join(feats)
    else:
        feat1, feat2 = selected_feats.columns
        feats = feat1 + ' \n + ' + feat2 + '\n ('+dat+')'

    m_errs[ind, :] = ind, feats, av_err_train, av_err_test

#%%
#Visualization

#create dataframe storing the errors and the corresponding features
err_df = pd.DataFrame(
    data = m_errs[:, 1:],
    index = m_errs[:, 0],
    columns = ['Features', 'Train Error', 'Test Error']
    )
#convert it to long format
err_df = pd.melt(
    err_df, id_vars = 'Features', 
    value_vars = ['Train Error', 'Test Error'], 
    value_name = 'Mean Squared Error',
    var_name = 'Error'
    )

#plot the results
plot = sns.catplot(
    data = err_df,
    x = 'Features',
    y = 'Mean Squared Error',
    hue = 'Error',
    kind = 'bar',
    height = 10,
    aspect = 2,
    palette = pal
    )
locs, labs = plt.xticks()
plt.setp(labs, rotation = 45, ha = 'right', fontsize = 30)
leg_title = plot._legend.get_title()
plt.setp(leg_title, fontsize = 38)
plt.ylim(bottom = 0.9)
plt.title('Averaged Train & Test Error for a Linear Regressor with different Feature Combinations')

#based on the results, serum sodium alone and serum sodium + age will be used further

#%%
#Regularization

y = data_norm[['SC']].values
X = rel_predictors.values

lambdas = np.logspace(-4, 6, 2000)
l_errs = np.zeros((len(lambdas), 2))
l_coefs = np.zeros((len(lambdas), 11))

#calculate weights and errors based on lambda
for i, l in enumerate(lambdas):
    cv = model_selection.KFold(n_splits = cvf, shuffle = True)
    errs = np.zeros((cvf, 2))
    coef_arr = np.zeros((cvf, 11))
    
    j = 0
    for train_i, test_i in cv.split(X):
        X_train, y_train = X[train_i, :], y[train_i]
        X_test, y_test = X[test_i, :], y[test_i]
        
        ridge = lm.Ridge(alpha = l)
        ridge = ridge.fit(X_train, y_train)

        y_pred1 = ridge.predict(X_train)
        y_pred2 = ridge.predict(X_test)
        
        err_train = np.square(y_train - y_pred1).sum()/len(y_train)
        err_test = np.square(y_test - y_pred2).sum()/len(y_test)
        errs[j, :] = err_train, err_test
        coefs = ridge.coef_.flatten()
        coef_arr[j, :] = coefs
        j += 1
    
    av_err_train, av_err_test = errs[:, 0].mean(), errs[:, 1].mean()
    l_errs[i, :] = av_err_train, av_err_test
    av_coefs = coef_arr.mean(axis = 0)
    l_coefs[i, :] = av_coefs
    
#fit a line to the test error for a better visualization
spline = UnivariateSpline(lambdas, l_errs[:, 1], s = 0.16)
test_est = spline(lambdas)
opt_ind, opt_lambda = np.argmin(test_est), np.min(test_est)
opt_ind = np.argmin(test_est) 
opt_lambda = lambdas[opt_ind]
#get the weights for the optimal lambda and store them with the associated features
opt_weights = l_coefs[opt_ind, :]
reg_feats = pd.DataFrame({
    'Attributes': rel_predictors.columns,
    'Optimal Weights': opt_weights
    })
reg_feats = reg_feats.sort_values(['Optimal Weights'], key = abs, ascending = False)
#again, serum sodium and age are the most optimal features
    
#%%
#Visualization 

#plot the results
fig = plt.figure(figsize = (30, 15))
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(lambdas, l_coefs)
ax1.set_xscale('log')
ax1.set(
       xlabel = r'Regularization Strength $\lambda$',
       ylabel = 'Averaged Weights'
       )
ax1.legend(labels = rel_predictors.columns, loc = 1, facecolor = 'white')
ax1.text(300000, -0.18, "A", fontsize = 55, c = "#6f6f6f")
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(lambdas, l_errs[:, 0], label = 'Train Error', c = '#9463A6')
ax2.plot(lambdas, l_errs[:, 1], label = 'Test Error', c = '#89A643')
ax2.plot(lambdas, test_est, label = 'Test Error \n (Estimate)', c = '#C24B36')
ax2.set_xscale('log')
ax2.legend(loc = 1, facecolor = 'white')
ax2.set(
       xlabel = r'Regularization Strength $\lambda$',
       ylabel = 'Mean Squared Error'
       )
ax2.text(300000, 0.917, "B", fontsize = 55, c = "#6f6f6f")
fig_title = fig.suptitle('Feature Weights and Train/Test Error of a Linear Regressor \n dependent on the Regularization Strength', fontsize = 40)
fig_title.set_position([.5, 0.96])

#%%
#Optimal Features Analysis - based on Regularization

#extract the cumulative combinations of the top 5 features according to the regularization
feat_l = []
for feat in reg_feats['Attributes'].values[:5]:
    if len(feat_l) == 0:
        feat_l.append([feat])
    elif len(feat_l) > 0:
        feat = [feat] + feat_l[-1]
        feat_l.append(feat)

y = data_norm[['SC']].values

#test the feature combinations
m_errs = np.empty((len(feat_l), 3), dtype = object)
for ind, comb in enumerate(feat_l):
    X = data_norm[comb].values
    cv = model_selection.KFold(n_splits = cvf, shuffle = True)
    errs = np.zeros((cvf, 2))
    
    i = 0
    for train_i, test_i in cv.split(X):
        X_train, y_train = X[train_i, :], y[train_i]
        X_test, y_test = X[test_i, :], y[test_i]

        regressor = lm.LinearRegression().fit(X_train, y_train)
        y_pred1 = regressor.predict(X_train)
        y_pred2 = regressor.predict(X_test)

        err_train = np.square(y_train - y_pred1).sum()/len(y_train)
        err_test = np.square(y_test - y_pred2).sum()/len(y_test)
        errs[i, :] = err_train, err_test
        i += 1
    
    av_err_train, av_err_test = errs[:, 0].mean(), errs[:, 1].mean()
    m_errs[ind, :] = len(comb), av_err_train, av_err_test
    
#%%
#Visualization

#create dataframe storing the errors and the corresponding features nubmers
reg_feats_ana = pd.DataFrame(
    data = m_errs,
    columns = ['Number of Attributes', 'Train Error', 'Test Error']
    )
#convert it to long format
reg_feats_ana = pd.melt(
    reg_feats_ana, id_vars = 'Number of Attributes', 
    value_vars = ['Train Error', 'Test Error'], 
    value_name = 'Mean Squared Error',
    var_name = 'Error'
    )

#plot the results
plot = sns.catplot(
    data = reg_feats_ana,
    x = 'Number of Attributes',
    y = 'Mean Squared Error',
    hue = 'Error',
    kind = 'bar',
    height = 10,
    aspect = 2,
    palette = pal
    )
plt.ylim(bottom = 0.90)
leg_title = plot._legend.get_title()
plt.setp(leg_title, fontsize = 38)
plt.title('Averaged Train & Test Error for a Linear Regressor with a different Number of \n the best Features selected via Regularization')

#%%
#Optimize Regression Model
y = reg1.values[:, 0]
X = reg1.values[:, 1:]
# =============================================================================
# 
# Xpow = np.power(X, 3)
# Xlog = np.log(abs(X))
# Xsqrt = np.sqrt(abs(X))
# Xcbrt = np.cbrt(X)
# X = np.hstack((X, np.power(X, 2)))
# X = np.hstack((X, Xlog))
# X = np.hstack((X, Xsqrt))
# X = np.hstack((X, Xcbrt))
# =============================================================================
#X = Xsq

cv = model_selection.KFold(n_splits = cvf, shuffle = True)
errs = np.zeros((cvf, 2))
i = 0
for train_i, test_i in cv.split(X):
    X_train, y_train = X[train_i, :], y[train_i]
    X_test, y_test = X[test_i, :], y[test_i]

    ridge = lm.Ridge(alpha = opt_lambda).fit(X_train, y_train)
    y_pred1 = ridge.predict(X_train)
    y_pred2 = ridge.predict(X_test)

    err_train = np.square(y_train - y_pred1).sum()/len(y_train)
    err_test = np.square(y_test - y_pred2).sum()/len(y_test)
    errs[i, :] = err_train, err_test
    i += 1

av_err_train, av_err_test = errs[:, 0].mean(), errs[:, 1].mean()
print(av_err_train, av_err_test)

# %%

columns = ["Age", "Anaemia", "CPK", "Diabetes", "EF", "hBP", "Platelets", "SC", "sNa", "Sex", "Smoking", "Time", "Death Event"] - ['SC','Death Event']
