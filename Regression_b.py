import numpy as np
import pandas as pd
#from scipy.interpolate import UnivariateSpline
from scipy.stats import t
from sklearn import model_selection
import sklearn.linear_model as lm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from plot_style import set_plot_theme
import toolbox_02450

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

#general parameters
in_cvf = 10

#%%
#Set up an ANN

#ann global parameters
epochs = 10
batch_size = 59

reg1 = data_reg[['SC', 'sNa']]
reg2 = data_reg[['SC', 'sNa', 'Age']]
#reg2 = data_norm[['SC']+[col for col in rel_predictors.columns]]

reg_l = [reg1, reg2]

#define the ANN function that creates a multi-layer perceptron (MLP) for linear regression
def ann_reg1(in_nodes = 2, hid_nodes = 2, out_nodes = 1):
    mlp = torch.nn.Sequential(
        torch.nn.AdaptiveMaxPool1d(in_nodes),
        torch.nn.Linear(in_nodes, hid_nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(hid_nodes, out_nodes)
        )
    return mlp

#the same principle, but now two hidden layers
def ann_reg2(in_nodes = 2, hid1_nodes = 2, hid2_nodes = 2, out_nodes = 1):
    mlp = torch.nn.Sequential(
        torch.nn.AdaptiveMaxPool1d(in_nodes),
        torch.nn.Linear(in_nodes, hid1_nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(hid1_nodes, hid2_nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(hid2_nodes, out_nodes)
        )
    return mlp

#3 hidden layers
def ann_reg3(in_nodes = 2, hid_nodes = 2, out_nodes = 1):
    mlp = torch.nn.Sequential(
        torch.nn.AdaptiveMaxPool1d(in_nodes),
        torch.nn.Linear(in_nodes, hid_nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(hid_nodes, hid_nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(hid_nodes, hid_nodes),
        torch.nn.Tanh(),
        torch.nn.Linear(hid_nodes, out_nodes)
        )
    return mlp

#define a batch function to let the ANN train on different batches of data, thus improving accuracy
def get_batch(X, y, size):
    for i in range(0, X.shape[0], size):
        yield (X[i:i + size], y[i:i + size])

#define the loss function for regression
loss_fun = torch.nn.MSELoss()

layers = 2
nodes = 3
#compare model performance with one and two features
m_errs = np.empty((len(reg_l)*layers*nodes, 5), dtype = object)
ind = 0
for dat in reg_l:
    y = dat.values[:, 0]
    y = np.reshape(y, (-1, 1))
    X = dat.values[:, 1:]
    
    #compare performance for 1 and 3 hidden layers
    for layer_len in range(1, layers+1):
        
        #compare performance for different number of hidden nodes
        for n_nodes in range(1, nodes+1):
            
            #perform 10-fold cross-validation
            cv = model_selection.KFold(n_splits = in_cvf, shuffle = True)
            errs = np.zeros((in_cvf, 2))
            #cross-validate ann models
            for f, (train_i, test_i) in enumerate(cv.split(X)):
                #print(f'Current CV-fold: {f+1}/{in_cvf}.')
                #print()
                X_train, y_train = X[train_i, :], y[train_i]
                X_test, y_test = X[test_i, :], y[test_i]
                
                #convert to tensors
                X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
                X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
                
                #initialize model
                if layer_len == 1:
                    nnet = ann_reg1(in_nodes = X.shape[1], hid_nodes = n_nodes)
                    for i in range(2 + 2 * 1):
                        if i % 2 != 0:
                            torch.nn.init.xavier_uniform_(nnet[i].weight)
                elif layer_len == 2:
                    #nnet = ann_reg2(in_nodes = X.shape[1], hid1_nodes = n_nodes, hid2_nodes = n_nodes)
                    nnet = ann_reg3(in_nodes = X.shape[1], hid_nodes = n_nodes)
                    for i in range(8):
                        if i % 2 != 0:
                            torch.nn.init.xavier_uniform_(nnet[i].weight)
                opt = torch.optim.Adam(nnet.parameters())
                #print(f'Initialize epochs: {epochs}.')
                #print()
                
                #run replicates
                for e in range(epochs):
                    #set model to training mode
                    #print('Training model.')
                    #print()
                    train_loss = 0
                    train_err = 0
                    train_samples = 0
                    
                    train_y_est_arr = np.empty((270, 1))
                    
                    nnet.train()
                    
                    #train model on batches of data
                    for X_batch, y_batch in get_batch(X_train, y_train, batch_size):
                        #X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        train_y_est = nnet(X_batch)
                        loss = loss_fun(train_y_est, y_batch)
                        
                        #train_y_est = train_y_est.detach().numpy()
                        np.vstack((train_y_est_arr, train_y_est.detach().numpy()))
                        
                        #back propagation
                        opt.zero_grad(); loss.backward(); opt.step()
                        
                        train_loss += loss.item() * y_batch.size(0)
                        train_err += (np.square(train_y_est.detach().numpy() - y_batch.detach().numpy()).sum()/len(train_y_est)).item()
                        train_samples += y_batch.size(0)
                        #print(f'Current epoch: {e+1}/{epochs}; loss: {train_loss/train_samples}; error: {train_err/train_samples}')
                    #print()
                    #print(f'Final training loss: {train_loss/train_samples}')
                    #print()
                    
                    #set model to testing (evaluating) mode
                    #print('Testing model.')
                    #print()
                    test_loss = 0
                    test_err = 0
                    test_samples = 0
                    
                    nnet.eval()
                    
                    #test model on batches of data
                    with torch.no_grad():
                        for X_batch, y_batch in get_batch(X_test, y_test, batch_size):
                            #X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                            test_y_est = nnet(X_batch)
                            loss = loss_fun(test_y_est, y_batch)
                            
                            test_loss += loss.item() * y_batch.size(0)
                            test_err += (np.square(test_y_est.detach().numpy() - y_batch.detach().numpy()).sum()/len(test_y_est)).item()
                            test_samples += y_batch.size(0)
                            #print(f'Current epoch: {e+1}/{epochs}; loss: {test_loss/test_samples}; error: {test_err/test_samples}')
                    #print()
                    #print(f'Final testing loss: {test_loss/test_samples}')
                    #print()
                errs[f, :] = train_err/train_samples, test_err
            
            av_err_train, av_err_test = errs[:, 0].mean(), errs[:, 1].mean()
            
            feats = dat.drop('SC', axis = 1).columns
            feats = [feat for feat in feats]
            feats = ' \n + '.join(feats)
            
            m_errs[ind, :] = feats, layer_len, n_nodes, av_err_train, av_err_test
            
            ind += 1
    
#%%
#Visualize Results of ANN Configuration

#create dataframe storing the errors, hiddel layer info and the corresponding features
err_df = pd.DataFrame(
    data = m_errs,
    columns = ['Features', 'No. of Hidden Layers', 'No. of Hidden Nodes', 'Train Error', 'Test Error']
    )
err_df1 = err_df.loc[err_df['No. of Hidden Layers'] == 1].drop('No. of Hidden Layers', axis = 1)
err_df2 = err_df.loc[err_df['No. of Hidden Layers'] == 2].drop('No. of Hidden Layers', axis = 1)

#convert it to long format
err_df1 = pd.melt(
    err_df1, 
    id_vars = ['No. of Hidden Nodes', 'Features'], 
    value_vars = ['Train Error', 'Test Error'], 
    value_name = 'Mean Squared Error',
    var_name = 'Error'
    )
err_df1['Nod_Feat'] = ['(' for i in range(12)] + err_df1['No. of Hidden Nodes'].astype(str) + [') ' for i in range(12)] + err_df1['Features']
err_df2 = pd.melt(
    err_df2, 
    id_vars = ['No. of Hidden Nodes', 'Features'], 
    value_vars = ['Train Error', 'Test Error'], 
    value_name = 'Mean Squared Error',
    var_name = 'Error'
    )
err_df2['Nod_Feat'] = ['(' for i in range(12)] + err_df2['No. of Hidden Nodes'].astype(str) + [') ' for i in range(12)] + err_df2['Features']

#plot the results
#1 hidden layer
plot = sns.catplot(
    data = err_df1,
    x = 'Nod_Feat',
    y = 'Mean Squared Error',
    hue = 'Error',
    kind = 'bar',
    height = 10,
    aspect = 2,
    palette = pal
    )
plt.title('Averaged Train & Test Error for an ANN with different Feature Combinations and Hidden Layer Nodes \n 1 Hidden Layer')
plt.xlabel('No. of hidden Nodes, Features')
locs, labs = plt.xticks()
plt.setp(labs, rotation = 45, ha = 'right', fontsize = 30)
leg_title = plot._legend.get_title()
plt.setp(leg_title, fontsize = 38)
#plt.ylim(top = 0.07)

#2 hidden layers
plot = sns.catplot(
    data = err_df2,
    x = 'Nod_Feat',
    y = 'Mean Squared Error',
    hue = 'Error',
    kind = 'bar',
    height = 10,
    aspect = 2,
    palette = pal
    )
plt.title('Averaged Train & Test Error for an ANN with different Feature Combinations and Hidden Layer Nodes \n 3 Hidden Layers')
plt.xlabel('No. of hidden Nodes, Features')
locs, labs = plt.xticks()
plt.setp(labs, rotation = 45, ha = 'right', fontsize = 30)
leg_title = plot._legend.get_title()
plt.setp(leg_title, fontsize = 38)
#plt.ylim(top = 0.07)
#%%
#Visualization of the ANN Regression - Element-wise

#training prediction
fig = plt.figure(figsize = (30, 12))
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(
    range(270),
    y_train.detach().numpy(),
    label = 'True Value',
    c = '#5028D1',
    s = 70
    )

ax1.scatter(
    range(270),
    train_y_est_arr,
    label = 'Est. Value',
    c = '#B87C48',
    s = 70
    )

ax1.set(
        title = 'Training Prediction',
        xlabel = 'Value Index',
        ylabel = 'SC Level (normalized)'
        )
ax1.legend(loc = 1, facecolor = 'white')

#testing prediction
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(
    range(29),
    y_test.detach().numpy(),
    label = 'True Value',
    c = '#5028D1',
    s = 70
    )

ax2.scatter(
    range(29),
    test_y_est,
    label = 'Est. Value',
    c = '#4FAB64',
    s = 70
    )

ax2.set(
        title = 'Testing Prediction',
        xlabel = 'Value Index',
        ylabel = 'SC Level (normalized)'
        )
ax2.legend(loc = 1, facecolor = 'white')
fig_title = fig.suptitle('Exemplary Visualization of the Difference between true and estimated Levels of SC \n with regards to the Training and Testing Set of an ANN', fontsize = 40)
fig_title.set_position([.5, 1.03])

#%%
#Comparison: Baseline vs. Linear Regression vs. ANN

#test on reg1
y = reg2.values[:, 0]
y = np.reshape(y, (-1, 1))
X = reg2.values[:, 1:]

#regression parameters
lambdas = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 5, 6, 7, 10, 12, 14])
l_errs = np.zeros((len(range(in_cvf)), 2))

#ann parameters
nodes = 10
h_errs = np.zeros((len(range(1, nodes+1)), 2))

out_cvf = 10
out_cv = model_selection.KFold(n_splits = out_cvf, shuffle = True)

comp_errs = np.empty((out_cvf, 5), dtype = object)
#outer cross validation
k = 0
for train_i, test_i in out_cv.split(X):
    X_train_out, y_train_out = X[train_i, :], y[train_i]
    X_test_out, y_test_out = X[test_i, :], y[test_i]
    
    #baseline
    #bas_y_est = np.array([y.mean() for y_i in y_test_out])
    bas_y_est = np.array([y_train_out.mean() for _ in y_test_out])
    bas_test_err = np.square(bas_y_est - y_test_out).sum()/len(y_test_out)
    
    #linear regression
    in_cv = model_selection.KFold(n_splits = in_cvf)#, shuffle = True)
    errs = np.zeros((len(lambdas), 2))
    
    #inner cross validation
    f = 0
    for train_j, test_j in in_cv.split(X_train_out):
        
        X_train_in, y_train_in = X_train_out[train_j, :], y_train_out[train_j]
        X_test_in, y_test_in = X_train_out[test_j, :], y_train_out[test_j]
        
        for i, l in enumerate(lambdas):
            ridge = lm.Ridge(alpha = l)
            ridge = ridge.fit(X_train_in, y_train_in)
        
            y_pred1 = ridge.predict(X_train_in)
            y_pred2 = ridge.predict(X_test_in)
                
            err_train = np.square(y_train_in - y_pred1).sum()/len(y_train_in)
            err_test = np.square(y_test_in - y_pred2).sum()/len(y_test_in)
            errs[i, :] = err_train, err_test
            
        opt_ind, l_err = np.argmin(errs[:, 1]), np.min(errs[:, 1])
        opt_lambda = lambdas[opt_ind]
        l_errs[f, :] = opt_lambda, l_err
            
        f += 1
    
    opt_lambda = l_errs[np.argmin(l_errs[:, 1]), 0]
    
    ridge = lm.Ridge(alpha = opt_lambda)
    ridge = ridge.fit(X_train_out, y_train_out)
    
    reg_y_est = ridge.predict(X_test_out)
    reg_test_err = np.square(reg_y_est - y_test_out).sum()/len(y_test_out)
    
    #ann
    X_train_out, y_train_out = torch.from_numpy(X_train_out).float(), torch.from_numpy(y_train_out).float()
    X_test_out, y_test_out = torch.from_numpy(X_test_out).float(), torch.from_numpy(y_test_out).float()
    
    #inner cross validation
    f = 0
    for train_i, test_i in in_cv.split(X_train_out):
        X_train_in, y_train_in = X_train_out[train_i, :], y_train_out[train_i]
        X_test_in, y_test_in = X_train_out[test_i, :], y_train_out[test_i]
        
        for i, n_nodes in enumerate(range(1, nodes+1)):
            #nnet = ann_reg2(in_nodes = 2, hid1_nodes = n_nodes, hid2_nodes = n_nodes)
            nnet = ann_reg3(in_nodes = 2, hid_nodes = n_nodes)
            torch.nn.init.xavier_uniform_(nnet[1].weight)
            torch.nn.init.xavier_uniform_(nnet[3].weight)
            opt = torch.optim.Adam(nnet.parameters())
            
            for e in range(epochs):
                train_loss = 0
                train_err = 0
                train_samples = 0
                    
                nnet.train()
                    
                for X_batch, y_batch in get_batch(X_train_in, y_train_in, batch_size):
                    y_est = nnet(X_batch)
                    loss = loss_fun(y_est, y_batch)
                        
                    opt.zero_grad(); loss.backward(); opt.step()
                        
                    train_loss += loss.item() * y_batch.size(0)
                    train_err += (np.square(y_est.detach().numpy() - y_batch.detach().numpy()).sum()/len(y_est)).item()
                    train_samples += y_batch.size(0)

                test_loss = 0
                test_err = 0
                test_samples = 0
                    
                nnet.eval()

                with torch.no_grad():
                    for X_batch, y_batch in get_batch(X_test_in, y_test_in, batch_size):
                        y_est = nnet(X_batch)
                        loss = loss_fun(y_est, y_batch)
                            
                        test_loss += loss.item() * y_batch.size(0)
                        test_err += (np.square(y_est.detach().numpy() - y_batch.detach().numpy()).sum()/len(y_est)).item()
                        test_samples += y_batch.size(0)
                    
            errs[i, :] = train_err/train_samples, test_err
            
        av_err_train, av_err_test = errs[:, 0].mean(), errs[:, 1].mean()
        h_errs[f, :] = av_err_train, av_err_test
        
        f += 1
    
    opt_h, opt_err = np.argmin(h_errs[:, 1])+1, np.min(h_errs[:, 1])
    
    #nnet = ann_reg2(in_nodes = 1, hid1_nodes = opt_h, hid2_nodes = opt_h)
    nnet = ann_reg3(in_nodes = 2, hid_nodes = n_nodes)
    torch.nn.init.xavier_uniform_(nnet[1].weight)
    torch.nn.init.xavier_uniform_(nnet[3].weight)
    opt = torch.optim.Adam(nnet.parameters())
    
    for e in range(epochs):
        train_loss = 0
        train_err = 0
        train_samples = 0
                    
        nnet.train()
                    
        for X_batch, y_batch in get_batch(X_train_out, y_train_out, batch_size):
            y_est = nnet(X_batch)
            loss = loss_fun(y_est, y_batch)
                        
            opt.zero_grad(); loss.backward(); opt.step()
                        
            train_loss += loss.item() * y_batch.size(0)
            train_err += (np.square(y_est.detach().numpy() - y_batch.detach().numpy()).sum()/len(y_est)).item()
            train_samples += y_batch.size(0)

        test_loss = 0
        test_err = 0
        test_samples = 0
        
        nnet.eval()

        with torch.no_grad():
            for X_batch, y_batch in get_batch(X_test_out, y_test_out, batch_size):
                y_est = nnet(X_batch)
                loss = loss_fun(y_est, y_batch)
                            
                test_loss += loss.item() * y_batch.size(0)
                test_err += (np.square(y_est.detach().numpy() - y_batch.detach().numpy()).sum()/len(y_est)).item()
                test_samples += y_batch.size(0)
                
    nn_test_err = test_err
    
    comp_errs[k, :] = opt_lambda, reg_test_err, opt_h, nn_test_err, bas_test_err
    
    k += 1

#%%
#store the results in a dataframe
err_tab = pd.DataFrame(
    data = comp_errs,
    columns = ['Regularization Strength', 'Regression Error', 'No. of hidden Nodes', 'ANN Error', 'Baseline Error']
    )
#err_tab.to_excel('../Data/table_reg-b-1.xlsx')

#%%
#Statistical Evaluation I: Generate Data

#define best hyperparameters
best_lambda = err_tab['Regularization Strength'].mean()
best_h = int(err_tab['No. of hidden Nodes'].mean())

#test on reg1
y = reg1.values[:, 0]
y = np.reshape(y, (-1, 1))
X = reg1.values[:, 1:]
Xlog, Xsqrt, Xcbrt = np.log(abs(X)), np.sqrt(abs(X)), np.cbrt(X)
X = np.hstack((X, Xlog, Xsqrt, Xcbrt))

#generate j splits and test the models
replicates = 10
cvf = 10
r = np.empty((len(range(replicates * cvf)), 3))
m_errs = np.empty((len(range(replicates * cvf)), 3))
est_arr = np.empty((len(range(replicates * cvf)), 30, 2))
j = 0
for rep in range(replicates):
    
    rand_cv = model_selection.ShuffleSplit(n_splits = cvf)
    
    for train_t, test_i in rand_cv.split(X):
        X_train, y_train = X[train_i, :], y[train_i]
        X_test, y_test = X[test_i, :], y[test_i]
        
        #baseline
        bas_y_est = np.array([y.mean() for y_i in y_test]).T
        bas_test_err = np.square(bas_y_est - y_test).sum()/len(y_test)
        
        #linear regression
        ridge = lm.Ridge(alpha = best_lambda)
        ridge = ridge.fit(X_train, y_train)
    
        reg_y_est = ridge.predict(X_test)
        reg_test_err = np.square(reg_y_est - y_test).sum()/len(y_test)
        
        #ann
        X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        
        nnet = ann_reg2(in_nodes = 1, hid1_nodes = best_h, hid2_nodes = best_h)
        torch.nn.init.xavier_uniform_(nnet[1].weight)
        torch.nn.init.xavier_uniform_(nnet[3].weight)
        opt = torch.optim.Adam(nnet.parameters())
        
        for e in range(epochs):
            train_loss = 0
            train_err = 0
            train_samples = 0
            nnet.train()
            for X_batch, y_batch in get_batch(X_train, y_train, batch_size):
                y_est = nnet(X_batch)
                loss = loss_fun(y_est, y_batch)
                opt.zero_grad(); loss.backward(); opt.step()
                train_loss += loss.item() * y_batch.size(0)
                train_err += (np.square(y_est.detach().numpy() - y_batch.detach().numpy()).sum()/len(y_est)).item()
                train_samples += y_batch.size(0)
            test_loss = 0
            test_err = 0
            test_samples = 0
            nnet.eval()
            with torch.no_grad():
                for X_batch, y_batch in get_batch(X_test, y_test, batch_size):
                    nn_y_est = nnet(X_batch)
                    loss = loss_fun(nn_y_est, y_batch)
                                
                    test_loss += loss.item() * y_batch.size(0)
                    test_err += (np.square(nn_y_est.detach().numpy() - y_batch.detach().numpy()).sum()/len(nn_y_est)).item()
                    test_samples += y_batch.size(0)
        nn_test_err = test_err
        nn_y_est = nn_y_est.detach().numpy()
        y_test = y_test.numpy()
        
        #compute differences
        d_bas_reg = np.mean(np.square(bas_y_est - y_test) - np.square(reg_y_est - y_test))
        d_bas_ann = np.mean(np.square(bas_y_est - y_test) - np.square(nn_y_est - y_test))
        d_reg_ann = np.mean(np.square(reg_y_est - y_test) - np.square(nn_y_est - y_test))
        
        r[j, :] = d_bas_reg, d_bas_ann, d_reg_ann
        m_errs[j, :] = bas_test_err, reg_test_err, nn_test_err
        est_arr[j, :, 0] = reg_y_est.flatten()
        est_arr[j, :, 1] = nn_y_est.flatten()
        
        j += 1
        
est_arr = np.mean(est_arr, axis = 0)

#%%
#Statistical Evaluation II: Use Data
#using the toolbox code to have it in this script directly

#evaluate statistically
rho = 1/cvf
alpha = 0.05
J = r.shape[0]

#baseline vs. regression
r_hat_bas_reg = np.mean(r[:, 0])
s_hat_bas_reg = np.var(r[:, 0])
sigma_tilde_bas_reg = ( 1/J + rho/(1 - rho) ) * s_hat_bas_reg

ci_bas_reg = t.interval(1 - alpha, df = J, loc = r_hat_bas_reg, scale = sigma_tilde_bas_reg)
p_val_bas_reg = 2 * t.cdf(-np.abs(r_hat_bas_reg)/sigma_tilde_bas_reg, df = J - 1)

#baseline vs. ann
r_hat_bas_ann = np.mean(r[:, 1])
s_hat_bas_ann = np.var(r[:, 1])
sigma_tilde_bas_ann = ( 1/J + rho/(1 - rho) ) * s_hat_bas_ann

ci_bas_ann = t.interval(1 - alpha, df = J, loc = r_hat_bas_ann, scale = sigma_tilde_bas_ann)
p_val_bas_ann = 2 * t.cdf(-np.abs(r_hat_bas_ann)/sigma_tilde_bas_ann, df = J - 1)

#regression vs. ann
r_hat_reg_ann = np.mean(r[:, 2])
s_hat_reg_ann = np.var(r[:, 2])
sigma_tilde_reg_ann = ( 1/J + rho/(1 - rho) ) * s_hat_reg_ann

ci_reg_ann = t.interval(1 - alpha, df = J, loc = r_hat_reg_ann, scale = sigma_tilde_reg_ann)
p_val_reg_ann = 2 * t.cdf(-np.abs(r_hat_reg_ann)/sigma_tilde_reg_ann, df = J - 1)

stat_df = pd.DataFrame({
    'Confidence Interval': [ci_bas_reg, ci_bas_ann, ci_reg_ann],
    'Test Statistic': [
        -np.abs(r_hat_bas_reg)/sigma_tilde_bas_reg, 
        -np.abs(r_hat_bas_ann)/sigma_tilde_bas_ann, 
        -np.abs(r_hat_reg_ann)/sigma_tilde_reg_ann
        ],
    'p value': [p_val_bas_reg, p_val_bas_ann, p_val_reg_ann]
    })

#%%
#Visualize element-wise Regression - Linear/Ridge and ANN

fig = plt.figure(figsize = (30, 12))
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(
    range(len(X_test.detach().numpy()[:, 0])),
    y_test,
    label = 'True Value',
    c = '#5028D1',
    s = 70
    )
ax1.scatter(
    range(len(X_test.detach().numpy()[:, 0])),
    est_arr[:, 0],
    label = 'Regressor Pred.',
    c = '#B87C48',
    s = 70
    )
ax1.set(
        title = 'Ridge Regression',
        xlabel = 'Value Index',
        ylabel = 'SC Level (normalized)'
        )
ax1.legend(loc = 1, facecolor = 'white')
#plt.ylim(top = 2)

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(
    range(len(X_test.detach().numpy()[:, 0])),
    y_test,
    label = 'True Value',
    c = '#5028D1',
    s = 70
    )
ax2.scatter(
    range(len(X_test.detach().numpy()[:, 0])),
    est_arr[:, 1],
    label = 'ANN Pred.',
    c = '#4FAB64',
    s = 70
    )
ax2.set(
        title = 'Artificial Neural Network',
        xlabel = 'Value Index',
        ylabel = 'SC Level (normalized)'
        )
ax2.legend(loc = 1, facecolor = 'white')
#plt.ylim(top = 2)
fig_title = fig.suptitle('Exemplary Visualization of the Difference between true and estimated Levels of SC \n predicted by the two ML Methods', fontsize = 40)
fig_title.set_position([.5, 1.03])


# %%
import toolbox_02450
results = []
for i in range(3):
    values = toolbox_02450.statistics.correlated_ttest(r[:,i], rho)
    result = {}
    result["p"] = values[0]
    result["CI"] = values[1]
    results.append(result)
    
print(results)
