"""
Predict sales prices and practice feature engineering, RFs, and gradient boosting
'House Prices: Advanced Regression Techniques' on Kaggle

Dataset is a featured challenge on Kaggle.
The Ames Housing dataset was compiled by Dean De Cock for use in data science education.
It's an incredible alternative for data scientists looking for a modernized and expanded
version of the often cited Boston Housing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt

train_data = pd.read_csv('house_price_data/train.csv')
test_data = pd.read_csv('house_price_data/test.csv')

# It is recommended by the author to drop with lving area greater than 4000, due to the irregular sales that occurred.
train_data = train_data[train_data.GrLivArea < 4000]

sale_price = train_data['SalePrice']
train_data.drop(['SalePrice', 'Id'], axis=1)
test_data.drop(['Id'], axis=1)

# Shows all columns that contain a null value for train and test sets.
print('Train set columns with null values: ')
print(train_data.columns[train_data.isnull().sum() > 0])
print('Test set columns with null values: ')
print(test_data.columns[test_data.isnull().sum() > 0])

'''
Train set columns with null values: 
Index(['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
       'MiscFeature'],
      dtype='object')
'''
# Fill null entries with appropriate value. Data description has what NA entries should be represented by.

# Frontage: NA should be represented by 0
train_data.loc[:, 'LotFrontage'] = train_data.loc[:, 'LotFrontage'].fillna(0)
# Alley: NA should be represented by None (No alley).
train_data.loc[:, 'Alley'] = train_data.loc[:, 'Alley'].fillna('None')
# Masonary Veener: NA should be represented by None for type and 0 for size.
train_data.loc[:, 'MasVnrType'] = train_data.loc[:, 'MasVnrType'].fillna('None')
train_data.loc[:, 'MasVnrArea'] = train_data.loc[:, 'MasVnrArea'].fillna(0)
# Basement: NA should be represented by None (no basement).
train_data.loc[:, 'BsmtQual'] = train_data.loc[:, 'BsmtQual'].fillna('None')
train_data.loc[:, 'BsmtCond'] = train_data.loc[:, 'BsmtCond'].fillna('None')
train_data.loc[:, 'BsmtExposure'] = train_data.loc[:, 'BsmtExposure'].fillna('None')
train_data.loc[:, 'BsmtFinType1'] = train_data.loc[:, 'BsmtFinType1'].fillna('None')
train_data.loc[:, 'BsmtFinType2'] = train_data.loc[:, 'BsmtFinType2'].fillna('None')
# Basement: NA should be represented by 0 for footage and baths.
train_data.loc[:, 'BsmtFinSF2'] = train_data.loc[:, 'BsmtFinSF2'].fillna(0)
train_data.loc[:, 'BsmtUnfSF'] = train_data.loc[:, 'BsmtUnfSF'].fillna(0)
train_data.loc[:, 'TotalBsmtSF'] = train_data.loc[:, 'TotalBsmtSF'].fillna(0)
train_data.loc[:, 'BsmtFullBath'] = train_data.loc[:, 'BsmtFullBath'].fillna(0)
train_data.loc[:, 'BsmtHalfBath'] = train_data.loc[:, 'BsmtHalfBath'].fillna(0)
# Electrical: Guess most standard electrical.
train_data.loc[:, 'Electrical'] = train_data.loc[:, 'Electrical'].fillna('SBrkr')
# Fireplace: NA represented by None (no fireplace).
train_data.loc[:, 'FireplaceQu'] = train_data.loc[:, 'FireplaceQu'].fillna('None')
# Garage: NA for Year should be 0 and None (No basement) for the rest.
train_data.loc[:, 'GarageType'] = train_data.loc[:, 'GarageType'].fillna('None')
train_data.loc[:, 'GarageYrBlt'] = train_data.loc[:, 'GarageYrBlt'].fillna(0)
train_data.loc[:, 'GarageFinish'] = train_data.loc[:, 'GarageFinish'].fillna('None')
train_data.loc[:, 'GarageQual'] = train_data.loc[:, 'GarageQual'].fillna('None')
train_data.loc[:, 'GarageCond'] = train_data.loc[:, 'GarageCond'].fillna('None')
# Pool: NA represented by None (No pool).
train_data.loc[:, 'PoolQC'] = train_data.loc[:, 'PoolQC'].fillna('None')
# Fence: NA represented by None (No fence).
train_data.loc[:, 'Fence'] = train_data.loc[:, 'Fence'].fillna('None')
# Misc Feature: NA represented by None (No misc features).
train_data.loc[:, 'MiscFeature'] = train_data.loc[:, 'MiscFeature'].fillna('None')

'''
Test set columns with null values: 
Index(['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
       'BsmtHalfBath', 'KitchenQual', 'Functional', 'FireplaceQu',
       'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
       'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType'],
      dtype='object')
'''

# Frontage: NA should be represented by 0
test_data.loc[:, 'LotFrontage'] = test_data.loc[:, 'LotFrontage'].fillna(0)
# Alley: NA should be represented by None (No alley).
test_data.loc[:, 'Alley'] = test_data.loc[:, 'Alley'].fillna('None')
# Utilities: NA represented by most common.
test_data.loc[:, 'Utilities'] = test_data.loc[:, 'Utilities'].fillna('AllPub')
# Exterior: NA represents None (as placeholder, unknown).
test_data.loc[:, 'Exterior1st'] = test_data.loc[:, 'Exterior1st'].fillna('Other')
test_data.loc[:, 'Exterior2nd'] = test_data.loc[:, 'Exterior2nd'].fillna('Other')
# Masonry veneer: NA should be represented by None for type and 0 for size.
test_data.loc[:, 'MasVnrType'] = test_data.loc[:, 'MasVnrType'].fillna('None')
test_data.loc[:, 'MasVnrArea'] = test_data.loc[:, 'MasVnrArea'].fillna(0)
# Basement: NA should be represented by None (no basement).
test_data.loc[:, 'BsmtQual'] = test_data.loc[:, 'BsmtQual'].fillna('None')
test_data.loc[:, 'BsmtCond'] = test_data.loc[:, 'BsmtCond'].fillna('None')
test_data.loc[:, 'BsmtExposure'] = test_data.loc[:, 'BsmtExposure'].fillna('None')
test_data.loc[:, 'BsmtFinType1'] = test_data.loc[:, 'BsmtFinType1'].fillna('None')
test_data.loc[:, 'BsmtFinType2'] = test_data.loc[:, 'BsmtFinType2'].fillna('None')
# Basement: NA should be represented by 0 for footage and bath numbers.
test_data.loc[:, 'BsmtFinSF1'] = test_data.loc[:, 'BsmtFinSF1'].fillna(0)
test_data.loc[:, 'BsmtFinSF2'] = test_data.loc[:, 'BsmtFinSF2'].fillna(0)
test_data.loc[:, 'BsmtUnfSF'] = test_data.loc[:, 'BsmtUnfSF'].fillna(0)
test_data.loc[:, 'TotalBsmtSF'] = test_data.loc[:, 'TotalBsmtSF'].fillna(0)
test_data.loc[:, 'BsmtFullBath'] = test_data.loc[:, 'BsmtFullBath'].fillna(0)
test_data.loc[:, 'BsmtHalfBath'] = test_data.loc[:, 'BsmtHalfBath'].fillna(0)
# Electrical: Guess most standard electrical.
test_data.loc[:, 'Electrical'] = test_data.loc[:, 'Electrical'].fillna('SBrkr')
test_data.loc[:, 'KitchenQual'] = test_data.loc[:, 'KitchenQual'].fillna('TA')
test_data.loc[:, 'Functional'] = test_data.loc[:, 'Functional'].fillna('Typ')
# Fireplace: NA represented by None (no fireplace).
test_data.loc[:, 'FireplaceQu'] = test_data.loc[:, 'FireplaceQu'].fillna('None')
# Garage: NA for Year should be 0 and None (No basement) for the rest.
test_data.loc[:, 'GarageType'] = test_data.loc[:, 'GarageType'].fillna('None')
test_data.loc[:, 'GarageYrBlt'] = test_data.loc[:, 'GarageYrBlt'].fillna(0)
test_data.loc[:, 'GarageFinish'] = test_data.loc[:, 'GarageFinish'].fillna('None')
test_data.loc[:, 'GarageCars'] = test_data.loc[:, 'GarageCars'].fillna(0)
test_data.loc[:, 'GarageArea'] = test_data.loc[:, 'GarageArea'].fillna(0)
test_data.loc[:, 'GarageQual'] = test_data.loc[:, 'GarageQual'].fillna('None')
test_data.loc[:, 'GarageCond'] = test_data.loc[:, 'GarageCond'].fillna('None')
# Pool: NA represented by None (No pool).
test_data.loc[:, 'PoolQC'] = test_data.loc[:, 'PoolQC'].fillna('None')
# Fence: NA represented by None (No fence).
test_data.loc[:, 'Fence'] = test_data.loc[:, 'Fence'].fillna('None')
# Misc Feature: NA represented by None (No Miscellaneous features)
test_data.loc[:, 'MiscFeature'] = test_data.loc[:, 'MiscFeature'].fillna('None')
test_data.loc[:, 'SaleType'] = test_data.loc[:, 'SaleType'].fillna('None')

'''
print('Train set columns with null values: ')
print(train_data.columns[train_data.isnull().sum() > 0])
print('Test set columns with null values: ')
print(test_data.columns[test_data.isnull().sum() > 0])
'''

categorical_features = train_data.select_dtypes(include = ["object"]).columns
numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
train_num = train_data[numerical_features]
train_cat = train_data[categorical_features]

print("Find most important features relative to target")
corr = train_data.corr()
corr.sort_values(["SalePrice"], ascending=False, inplace=True)
print(corr.SalePrice)

# One hot encode all categorical data for the test and train sets, int64 makes computation quicker versus strings.
encoded_train_data = pd.get_dummies(train_data)
encoded_test_data = pd.get_dummies(test_data)

X_train, X_test, y_train, y_test = train_test_split(encoded_train_data, sale_price, test_size=0.3, random_state=0)

stdSc = StandardScaler()
X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])
X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

y_train_pred = linear_reg.predict(X_train)
y_test_pred = linear_reg.predict(X_test)

'''
# Plot residuals
plt.scatter(y_train_pred, y_train_pred - y_train, c="blue", marker="s", label="Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c="lightgreen", marker="s", label="Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
plt.show()

# Plot predictions
plt.scatter(y_train_pred, y_train, c="blue", marker="s", label="Training data")
plt.scatter(y_test_pred, y_test, c="lightgreen", marker="s", label="Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc="upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
plt.show()
'''

scorer = make_scorer(mean_squared_error, greater_is_better=False)


def rmse_cv_train(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring=scorer, cv=10))
    return (rmse)


def rmse_cv_test(model):
    rmse = np.sqrt(-cross_val_score(model, X_test, y_test, scoring=scorer, cv=10))
    return (rmse)


print('Training RMSE: ', rmse_cv_train(linear_reg).mean())
print('Testing RMSE: ', rmse_cv_test(linear_reg).mean())

'''
All columns are now either float64(3), int64(35), or uint8(252), there are no more object dtype columns.

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Columns: 290 entries, Id to SaleCondition_Partial
dtypes: float64(3), int64(35), uint8(252)
memory usage: 792.8 KB
None
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1459 entries, 0 to 1458
Columns: 271 entries, Id to SaleCondition_Partial
dtypes: float64(11), int64(26), uint8(234)
memory usage: 755.2 KB
None
'''

'''
ridge = RidgeCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                        alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                        alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],
                cv=10)
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())
y_train_rdg = ridge.predict(X_train)
y_test_rdg = ridge.predict(X_test)

# Plot residuals
plt.scatter(y_train_rdg, y_train_rdg - y_train, c="blue", marker="s", label="Training data")
plt.scatter(y_test_rdg, y_test_rdg - y_test, c="lightgreen", marker="s", label="Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
plt.show()

# Plot predictions
plt.scatter(y_train_rdg, y_train, c="blue", marker="s", label="Training data")
plt.scatter(y_test_rdg, y_test, c="lightgreen", marker="s", label="Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc="upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
plt.show()

# Plot important coefficients
coefs = pd.Series(ridge.coef_, index=X_train.columns)
print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " + \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                       coefs.sort_values().tail(10)])
imp_coefs.plot(kind="barh")
plt.title("Coefficients in the Ridge Model")
plt.show()
'''
