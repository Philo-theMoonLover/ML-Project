import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn import linear_model
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

np.random.seed(11)

# https://www.kaggle.com/datasets/shivam2503/diamonds
datas = pd.read_csv('Diamond-Price-Prediction.csv')
s = (datas.dtypes == "object")
object_cols = list(s[s].index)

# Make copy to avoid changing original data
# label_data = datas.copy()
# # Apply label encoder to each column with categorical data
# label_encoder = LabelEncoder()
# for col in object_cols:
#     label_data[col] = label_encoder.fit_transform(label_data[col])

# correlation matrix
# cmap = sns.diverging_palette(70,20,s=50, l=40, n=6,as_cmap=True)
# corrmat= label_data.corr()
# f, ax = plt.subplots(figsize=(12,12))
# sns.heatmap(corrmat,cmap=cmap,annot=True)

# Drop rows contain zero-value
datas = datas.drop(datas[datas['Carat'] == 0].index)
datas = datas.drop(datas[datas['Cut'] == 0].index)
datas = datas.drop(datas[datas['Color'] == 0].index)
datas = datas.drop(datas[datas['Clarity'] == 0].index)
datas = datas.drop(datas[datas['Depth'] == 0].index)
datas = datas.drop(datas[datas['Table'] == 0].index)
datas = datas.drop(datas[datas['Price'] == 0].index)
datas = datas.drop(datas[datas['X'] == 0].index)
datas = datas.drop(datas[datas['Y'] == 0].index)
datas = datas.drop(datas[datas['Z'] == 0].index)

# plt.figure(figsize=(20,10))
# plt.subplot(1,2,1)
# plt.title('Price')
# sns.distplot(datas.Price/100)
# plt.subplot(1,2,2)
# plt.title('Price')
# sns.boxplot(x=datas.Price/100)
# plt.show()

datas.dropna(inplace=True)

print(datas.Price.describe(percentiles=[0.25, 0.5, 0.75, 0.85, 0.90, 1]))

# plt.figure(figsize=(35, 15))
# plt.subplot(1,3,1)
# plt1 = datas.Cut.value_counts().plot(kind = 'bar', color='c')
# plt.title('Cut Histogram')
# plt1.set(xlabel = 'Cut', ylabel = 'Frequency of Cut')
# plt.subplot(1,3,2)
# plt1 = datas.Color.value_counts().plot(kind = 'bar', color='c')
# plt.title('Color Histogram')
# plt1.set(xlabel = 'Color', ylabel = 'Frequency of Color')
# plt.subplot(1,3,3)
# plt1 = datas.Cut.value_counts().plot(kind = 'bar', color='c')
# plt.title('Clarity Histogram')
# plt1.set(xlabel = 'Clarity', ylabel = 'Frequency of Clarity')
# plt.show()

# ax = sns.regplot(x='Price',y='Z', data=datas, fit_reg=True, scatter_kws={"color":"#a9a799"}, line_kws={"color": "#835656"})
# ax.set_title(("Regression Line on Price vs 'y'",colors=="#4e4c39"))
# plt.show()

# Dropping the outliers
datas = datas[(datas['X'] > 2) & (datas['X'] < 9.5)]
datas = datas[(datas['Y'] < 10)]
datas = datas[(datas['Z'] > 1) & (datas['Z'] < 10)]

data_lr = datas[['Carat', 'Cut', 'Color', 'Clarity', 'Depth', 'Table', 'Price', 'X', 'Y', 'Z']]


# Defining the map function
def dummies(x, data):
    temp = pd.get_dummies(data[x], drop_first=True)
    data = pd.concat([data, temp], axis=1)
    data.drop([x], axis=1, inplace=True)
    return data


# Applying the function to the cars_lr
data_lr = dummies('Clarity', data_lr)
data_lr = dummies('Color', data_lr)
data_lr = dummies('Cut', data_lr)

# Label encoding the data
# Make copy to avoid changing original data
lb_data = datas.copy()
# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
# for col in object_cols:
#   lb_data[col] = label_encoder.fit_transform(lb_data[col])
lb_data['Cut'] = label_encoder.fit_transform(lb_data['Cut'])
lb_data['Color'] = label_encoder.fit_transform(lb_data['Color'])
lb_data['Clarity'] = label_encoder.fit_transform(lb_data['Clarity'])

''' Train_Test Split and feature scaling '''

data_train, data_test = train_test_split(data_lr, train_size=0.8, test_size=0.2, random_state=100)
lb_train, lb_test = train_test_split(lb_data, train_size=0.8, test_size=0.2, random_state=100)

# Dividing data into X and y variables to train dummy
yd_train = data_train.pop('Price')
Xd_train = data_train.copy()
# label
yl_train = lb_train.pop('Price')
Xl_train = lb_train

# Dividing data into X and y variables to test dummy
yd_test = data_test.pop('Price')
Xd_test = data_test.copy()
# label
yl_test = lb_test.pop('Price')
Xl_test = lb_test

''' Model Building '''

def plot_Learning_curve(model, x_train, y_train, type_variable):
    # Plot learning curve
    pipeline = make_pipeline(StandardScaler(), model)
    # Use learning curve to get training and test scores along with train sizes
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipeline, X=x_train, y=y_train,
                                                            cv=10, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    # Plot
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--',
             label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title('Learning Curve ' + type_variable)
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()


def print_metrics(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("R^2:", r2_score(y_test, y_pred))
    print("Adjusted R^2:",
          1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print('---------------------')


# Simple linear regression for dummy
lmd = linear_model.LinearRegression()
lmd.fit(Xd_train, yd_train)
# Simple linear regression for label
lml = lm_d = linear_model.LinearRegression()
lml.fit(Xl_train, yl_train)
print('---------------------')

# Root mean square error of dummy
data_rmse = np.sqrt(mean_squared_error(yd_test.to_numpy(), lmd.predict(Xd_test)))
print("Root Mean Square Error of 'dummy' variable:", data_rmse)
# Root mean square error of label
lb_rmse = np.sqrt(mean_squared_error(yl_test.to_numpy(), lml.predict(Xl_test)))
print("Root Mean Square Error of 'label' variable:", lb_rmse)

plot_Learning_curve(lmd, Xd_train, yd_train, 'with Dummy')
plot_Learning_curve(lml, Xl_train, yl_train, 'with Label')

print('Linear Regression with Dummy ---------------------')
# Linear regression with dummy
print_metrics(lmd, Xd_test, yd_test)
print('Linear Regression with Label Encoding ---------------------')
# linear regression with label
print_metrics(lml, Xl_test, yl_test)


# Polynomial regression
plm = linear_model.LinearRegression()
poly = PolynomialFeatures(2)
Xp_train = poly.fit_transform(Xd_train)
yp_train = yd_train
Xp_test = poly.fit_transform(Xd_test)
yp_test = yd_test
plm.fit(Xp_train, yp_train)

print("Polynomial Regression ---------------------")
print_metrics(plm, Xp_test, yp_test)
# plot_Learning_curve(plm, Xp_train, yp_train, 'Polynomial Regression')


''' OVERFITTING '''
print("L2 Regularization ---------------------")
ridge = Ridge(alpha=10)
# lasso = Lasso(alpha=10)

ridge.fit(Xd_train, yd_train)
# lasso.fit(Xd_train, yd_train)

print_metrics(ridge, Xd_test, yd_test)
# print_metrics(lasso, Xd_test, yd_test)

plot_Learning_curve(ridge, Xd_train, yd_train, 'with Ridge')
# plot_Learning_curve(lasso, Xd_train, yd_train, 'with Lasso')
