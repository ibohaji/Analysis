import pandas as pd
from scipy.stats import shapiro, spearmanr, kendalltau, mannwhitneyu, chi2_contingency
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_style import set_plot_theme, font
from cb91visuals import *

#!adjust path accordingly!
data_path = "Reports/Data/"
data = pd.read_csv(data_path+"heart_failure_clinical_records_dataset.csv", delimiter=',')
data_df = data

features = data_df.columns.copy()
labels_names = ['safe', 'dead']
labels = data_df["DEATH_EVENT"].astype(int).copy()

#Setting the default theme for the plots
set_plot_theme()

#change the colnames and value names for better clarity
data_df = data_df.set_axis(
    ["Age", "Anaemia", "CPK", "Diabetes", "EF", "hBP", "Platelets", "SC", "sNa", "Sex", "Smoking", "Time", "Death Event"],
    axis = 1
    )
measures = ["years","1000 mcg/L","Percentage","1000 kiloplatelets/mL","mg/dL","mEq/L", "Days"]

# %% Correlation

#Data Normalization

columns_to_standardize = [
    "Age","CPK", "EF", "Platelets", "SC", "sNa", "Time"
    ]

data_df_norm = data_df.copy()

for column in data_df.columns:
    data_df_norm[column] = (data_df[column]-data_df[column].mean())/data_df[column].std()

#quick check for missing values
missing = data.isnull().values.any()
#print(missing)
#there are none, as missing is 'False'

#remove binary data
bincols = data.columns[data.isin([0,1]).all()]
non_bincols = [col for col in data.columns if col not in bincols]
data_cont = data[non_bincols]
#data_cont.to_csv(data_path+"heart_failure_data__continuous-only.csv")

#data_blood = data_cont[["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "ejection_fraction", "platelets"]] 
#data_other = data_cont[["age", "time"]]

#test for normality
shapiro_df = pd.DataFrame(
    columns = ["Parameter", "Test Statistic", "p-value"]
    )

for col in data.columns:
    stat, pval = shapiro(data[col])
    test = pd.DataFrame(
        [[col, stat, pval]],
        columns = ["Parameter", "Test Statistic", "p-value"]
        )
    shapiro_df = shapiro_df.append(test, ignore_index = True)
#it seems that none of the parameters are normally distributed

#test for normalized data
norm_data = (data - data.mean())/data.std()
shapiro_norm_df = pd.DataFrame(
    columns = ["Parameter", "Test Statistic", "p-value"]
    )
for col in norm_data.columns:
    stat, pval = shapiro(norm_data[col])
    test = pd.DataFrame(
        [[col, stat, pval]],
        columns = ["Parameter", "Test Statistic", "p-value"]
        )
    shapiro_norm_df = shapiro_norm_df.append(test, ignore_index = True)
#same results


#test correlation
#since none of the parameters are normally distributed, Pearson correlation cannot be performed
#alternatives for non-normally distributed data are Spearman's rho and Kendall's tau
corr_df = pd.DataFrame(
    columns = ["Par1", "Par2", "Spearman's Rho", "Spearman Coef.", "Kendall's Tau", "Kendall Coef."]
    )
for col_pair in itertools.combinations(data.columns, 2):
    par1, par2 = col_pair
    #comb_name = par1 + " and " + par2
    rho_stat, rho_pval = spearmanr(data[par1], data[par2])
    tau_stat, tau_pval = kendalltau(data[par1], data[par2])
    sub_df = pd.DataFrame(
        [[par1, par2, rho_pval, rho_stat, tau_pval, tau_stat]],
        columns = ["Par1", "Par2", "Spearman's Rho", "Spearman Coef.", "Kendall's Tau", "Kendall Coef."]
        )
    corr_df = corr_df.append(sub_df)
corr_df = corr_df.sort_values("Spearman's Rho")
#interestingly, serum creatinine and CPK are not really correlated
corr_df_sig = corr_df[corr_df["Spearman's Rho"] <= 0.05]
#replicating results of Chicco and Jurman (2020) and selecting interesting correlations:
# - serum creatinine and sodium
# - age and serum creatinine
# - ejection fraction and serum creatinine

###look for correlations related to EF
ef_corr_df = corr_df[corr_df["Par1"] == "ejection_fraction"]
#the interesting ones are between serum_sodium, smoking, platelets and high blood pressure

#test other relationships between all other variables, as well as between serum creatinine
stat_df = pd.DataFrame(
    columns = ["Par1", "Par2", "M-W-U pVal", "chi-sq pVal"]
    )
for col_pair in itertools.combinations(data.columns, 2):
    par1, par2 = col_pair
    mwu_stat, mwu_pval = mannwhitneyu(data[par1], data[par2])

    conting_tab = pd.crosstab(data[par1], data[par2])
    chi2_stat, chi2_pval, chi2_dof, chi2_exp = chi2_contingency(conting_tab)
    sub_df = pd.DataFrame(
        [[par1, par2, mwu_pval, chi2_pval]],
        columns = ["Par1", "Par2", "M-W-U pVal", "chi-sq pVal"]
        )
    stat_df = stat_df.append(sub_df)
stat_df = stat_df.sort_values("chi-sq pVal")
#again, reproduced results (although p-values are still different)
#slice the dataframe to get (significant) relationships between serum creatinine or EF and the other attributes
int_stat_df = stat_df.loc[(stat_df["Par1"] == "serum_creatinine") | (stat_df["Par1"] == "ejection_fraction")]

#Plot a heatmap to see the correlation among the features
heatmap = sns.heatmap(data_df.corr(), vmin=-1, vmax=1, annot=True, cmap='Blues', fmt='.2f')
locs, labs = plt.yticks()
plt.setp(labs, fontsize = 15)
locs, labs = plt.xticks()
plt.setp(labs, rotation=45, ha = "right", fontsize = 15)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Coefficients', fontdict={'fontsize':23}, pad=12)

#plot correlation between sodium and creatinine only
fig = plt.figure(figsize = (15, 10))
ax = fig.add_subplot()
sns.regplot(data = data, x = "serum_sodium", y = "serum_creatinine")
ax.set(
       title = "Basic linear regression",
       xlabel = "SNa - mEq/L",
       ylabel = "mg/dL"
       )

#%% Data Normalization

columns_to_standardize = [
    "Age","CPK", "EF", "Platelets", "SC", "sNa", "Time"
    ]

data_df_norm = data_df.copy()

for column in data_df.columns:
    data_df_norm[column] = (data_df[column]-data_df[column].mean())/data_df[column].std()

#convert to array
X = data_df_norm.to_numpy()

#%% 3.2 Principal Component Analysis

# Run PCA algorithm
U, S, VT = np.linalg.svd(X, full_matrices=False)
rho = rho = (S*S) / (S*S).sum() 
threshold = 0.95

fig = plt.figure(figsize = (20, 10))
ax = fig.add_subplot()
ax.plot(range(1,len(rho)+1),rho,'x-', markersize = 20, color=CB91_Red)
ax.plot(range(1,len(rho)+1),np.cumsum(rho),'o-', markersize = 20, color=CB91_Green)
ax.plot([1,len(rho)],[threshold, threshold],'k--')
ax.set(
       title = 'Variance explained by Principal Components',
       xlabel = 'Number of Prinicipal Components',
       ylabel = 'Variance explained'
       )
ax.legend(['Individual','Cumulative','95% Threshold'])
plt.show()


#project the original dataset on the new subspace
X_tilde = X @ VT.T

# select the values projected on the first principal component
pc1 = X_tilde[:, 0]
# select the values projected on the second principal component
pc2 = X_tilde[:, 1]
# select the values projected on the third principal component
pc2 = X_tilde[:, 2]


#%% 2-D plot
fig = plt.figure(figsize = (11, 7))
ax = fig.add_subplot()
ax.scatter(pc1[labels==0], pc2[labels==0], label ="safe", marker='o', color = CB91_Green, edgecolors='black', linewidths=1)
ax.scatter(pc1[labels==1], pc2[labels==1], label="dead", marker='o', color = CB91_Red, edgecolors='black', linewidth=1)
ax.set(
       title = 'Projection on the First Two Principal Components',
       xlabel = 'PC 1',
       ylabel = 'PC 2',
       )
ax.legend(loc = 1, )
plt.show()

#%% Bar plot to see how weights of the original features in the new features space
fig, ax = plt.subplots(1,2,figsize=(16,8))
for i in range(2):
    ax[i].bar(data_df_norm.columns,VT.T[:,i])
    ax[i].set_xticklabels(data_df_norm.columns,rotation = 90)
    ax[i].set_title(f"{i+1}° Principal Component", fontsize = 24)
    ax[i].axhline(0)
    ax[i].set_ylim(-0.6,0.6)
    labs = ax[i].get_xticklabels()
    ax[i].set_xticklabels(labs, rotation = 45, ha = "right", fontsize = 20)
fig_title = fig.suptitle(
    "Weights of the First Two Principal Components", fontsize = 32
    )
fig_title.set_position([.5, 1.02])

# %% Attempt to make a classification based on the first principal component values
plt.figure()
labels_predicted = np.empty(299).astype(int)
labels_predicted[pc1>=-0.7] = 0
labels_predicted[pc1<-0.7] = 1

true_0  = np.sum(np.logical_and(data['DEATH_EVENT'] == 0, labels_predicted == 0))
false_0 = np.sum(np.logical_and(data['DEATH_EVENT'] == 1, labels_predicted == 0))
true_1  = np.sum(np.logical_and(data['DEATH_EVENT'] == 1, labels_predicted == 1))
false_1 = np.sum(np.logical_and(data['DEATH_EVENT'] == 0, labels_predicted == 1))

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


group_names = ['True Negative','False Positive','False Negative','True Positive']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix', size=28);
ax.set_xlabel('\nPredicted Values',size=18)
ax.set_ylabel('Actual Values',size=18);

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'],size=16)
ax.yaxis.set_ticklabels(['False','True'],size=16)

## Display the visualization of the Confusion Matrix.
plt.show()
