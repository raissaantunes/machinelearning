#%%
import pandas
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

df = pandas.read_excel('Time domian features based on structures.xlsx',
    sheet_name="Features",
    header=[0,1,2]
)

idx = pandas.IndexSlice
# df.loc[:, idx['Support Structure', :, 'Mean']].plot(figsize=(16,16))

# %%
# ##### Preprocessing data: Feature Scaling
# # import numpy as np
from sklearn import preprocessing

x = df.iloc[:, 1:]
preprocessing.normalize(x)

#%%

#choosing the right number of dimensions
from sklearn.decomposition import PCA

# x = df.iloc[:, 1:]
# y = pandas.Series(df.iloc[:,0], name=\"target\")
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# x_norm = preprocessing.normalize(x_train)

# pca = PCA()
# pca.fit(x_train)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
# d
# pca = PCA(n_components=30)
# x_reduced = pca.fit_transform(x_train)
# principalDf = pandas.DataFrame(
#     data=x_reduced,
#     columns=['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10',
#             'pca11', 'pca12', 'pca13', 'pca14', 'pca15', 'pca16', 'pca17', 'pca18', 'pca19', 'pca20',
#             'pca21', 'pca22', 'pca23', 'pca24', 'pca25', 'pca26', 'pca27', 'pca28', 'pca29', 'pca30']
# )
# df_new = pandas.concat([y_train, principalDf], axis=1)
# df_new

#%%

# Spliting data: 80% for train and 20% for test
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, train_size= 0.8)

#Listing columns to serve as index 
structure_list = ['Support Structure', 'Bottom Surface', '1st Half of hollow structure', '2nd Half of hollow structure', 'Bottom surface']
predictor_list = ['Mean', 'Variance', 'Max', 'Min']



#%%
##### Random Forest
#Hyperparameters: n_estimators (number of trees) = 100
from sklearn.ensemble import RandomForestRegressor
#fitting random forest to thw whole data
regressor = RandomForestRegressor(n_estimators=200, max_depth= 2) #tuning parameter of random forest, number of trees (n_estimator) = 250???

#%%

rf_relative_error = []

for i, s in enumerate(structure_list):
    average_relative_error = 0
    for p in predictor_list:
        predictors_cols = idx[structure_list[:i+1], :, p]
        regressor.fit(train.loc[:, predictors_cols].values, train.iloc[:, 0].values)
        y_hat_rf = regressor.predict(test.loc[:, predictors_cols])
        # fig, ax = plt.subplots()
        # ax.plot(test.iloc[:, 0].values, label='Observed')
        # ax.plot(y_hat_rf, label='Predicted')
        relative_error = abs((y_hat_rf - test.iloc[:, 0].values)/test.iloc[:, 0].values)
        average_relative_error += relative_error.mean()
    rf_relative_error.append(average_relative_error/5)

fig, ax = plt.subplots()
ax.plot(rf_relative_error)


# %%
##### SVR prediction
#Hyperparameters:
#kernel distribution= 'linear'
#Regularization C = 100
#Stopping Criteria E= 0.1

from sklearn.svm import SVR
svr_rbf_reg = SVR(kernel= 'rbf', C=100, epsilon=0.01) # the best tuning parameter is C=10^-11 and e = 0.1 ???
svr_relative_error = []

# %%
for i, s in enumerate(structure_list):
    average_relative_error = 0
    for p in predictor_list:
        predictors_cols = idx[structure_list[:i+1], :, p]
        svr_rbf_reg.fit(train.loc[:, predictors_cols].values, train.iloc[:, 0].values)
        y_hat_rbf = svr_rbf_reg.predict(test.loc[:, predictors_cols])
        # fig, ax = plt.subplots()
        # ax.plot(test.iloc[:, 0].values, label='Observed')
        # ax.plot(y_hat_rbf, label='Predicted')
        relative_error = abs((y_hat_rbf - test.iloc[:, 0].values)/test.iloc[:, 0].values)
        average_relative_error += relative_error.mean()
    svr_relative_error.append(average_relative_error/5)

fig, ax = plt.subplots()
ax.plot(svr_relative_error)


#%%

##### Ridge Regression 
# Hyperparameter:
# alpha=100 is the parameter that balances the amount emphasis given to min RSS.
#if alpha=0 the obj become same as simple linear regression

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=100, solver='cholesky') #best parameter tuning = 12500 is it alpha???
ridge_relative_error = []

# %%
for i, s in enumerate(structure_list):
    average_relative_error = 0
    for p in predictor_list:
        predictors_cols = idx[structure_list[:i+1], :, p]
        ridge_reg.fit(train.loc[:, predictors_cols], train.iloc[:, 0])
        y_hat_ridge = ridge_reg.predict(test.loc[:, predictors_cols])
        relative_error = abs((y_hat_ridge - test.iloc[:, 0].values)/test.iloc[:, 0].values)
        average_relative_error += relative_error.mean()
    ridge_relative_error.append(average_relative_error/5)

fig, ax = plt.subplots()
ax.plot(ridge_relative_error)



# %%
##### LASSO
#Hyperparameter:
#Step size = alpha = 10

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=10) #best tuning parameter is 10
lasso_relative_error = []

# %%
for i, s in enumerate(structure_list):
    average_relative_error = 0
    for p in predictor_list:
        predictors_cols = idx[structure_list[:i+1], :, p]
        lasso_reg.fit(train.loc[:, predictors_cols], train.iloc[:, 0])
        lasso_y_hat = lasso_reg.predict(test.loc[:, predictors_cols])
        relative_error = abs((lasso_y_hat - test.iloc[:, 0].values)/test.iloc[:, 0].values)
        average_relative_error += relative_error.mean()
    lasso_relative_error.append(average_relative_error/5)

fig, ax = plt.subplots()
ax.plot(lasso_relative_error)
# %%