#!/usr/bin/env python
# coding: utf-8

# ## 1. Credit card applications Commercial banks receive a lot< of applications for credit cards. Many of them get
# rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's
# credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and
# time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every
# commercial bank does so nowadays. In this notebook, we will build an automatic credit card approval predictor using
# machine learning techniques, just like the real banks do.

# The structure of this notebook is as follows:

# First, we will start off by loading and viewing the dataset.
# We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
# We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
# After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
# Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.
# First, loading and viewing the dataset. We find that since this data is confidential, the contributor of the dataset has anonymized the feature names.

# Import pandas
import pandas as pd

# Load dataset
cc_apps = pd.read_csv("datasets/cc_approvals.data", header=None)

# Inspect data
cc_apps.head()

# ## 2. Inspecting the applications The output may appear a bit confusing at its first sight, but let's try to figure
# out the most important features of a credit card application. The features of this dataset have been anonymized to
# protect the privacy. As we can see from our first glance
# at the data, the dataset has a mixture of numerical and non-numerical features. This can be fixed with some
# preprocessing, but before we do that, let's learn about the dataset a bit more to see if there are other dataset
# issues that need to be fixed.


# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print('\n')

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print('\n')

# Inspect missing values in the dataset
cc_apps.tail(17)



# ## 3. Splitting the dataset into train and test sets Now, we will split our data into train set and test set to
# prepare our data for two different phases of machine learning modeling: training and testing. Ideally,
# no information from the test data should be used to preprocess the training data or should be used to direct the
# training process of a machine learning model. Hence, we first split the data and then preprocess it.

# Import train_test_split
from sklearn.model_selection import train_test_split

# Drop the features 11 and 13
cc_apps = cc_apps.drop([11, 13], axis=1)

# Split into train and test sets
cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)

# ## 4. Handling the missing values (part i)
# Now we've split our data, we can handle some issues we identified when inspecting the DataFrame
# Now, let's temporarily replace these missing value question marks with NaN.

# Import numpy
import numpy as np

# Replace the '?'s with NaN in the train and test sets
cc_apps_train = cc_apps_train.replace('?', np.NaN)
cc_apps_test = cc_apps_test.replace('?', np.NaN)


# ## 5. Handling the missing values (part ii) We replaced all the question marks with NaNs. This is going to help us
# in the next missing value treatment that we are going to perform. An important question that gets raised here is
# why are we giving so much importance to missing values? Can't they be just ignored? Ignoring missing values can
# affect the performance of a machine learning model heavily. While ignoring the missing values our machine learning
# model may miss out on information about the dataset that may be useful for its training. Then, there are many
# models which cannot handle missing values implicitly such as Linear Discriminant Analysis (LDA). So,
# to avoid this problem, we are going to impute the missing values with a strategy called mean imputation.


# Impute the missing values with mean imputation
cc_apps_train.fillna(cc_apps_train.mean(), inplace=True)
cc_apps_test.fillna(cc_apps_train.mean(), inplace=True)

# Count the number of NaNs in the datasets and print the counts to verify
print(cc_apps_train.isnull().sum())
print(cc_apps_test.isnull().sum())


# ## 6. Handling the missing values (part iii) We have successfully taken care of the missing values present in the
# numeric columns. There are still some missing values to be imputed for columns 0, 1, 3, 4, 5, 6 and 13. All of
# these columns contain non-numeric data and this is why the mean imputation strategy would not work here. This needs
# a different treatment.

# Iterate over each column of cc_apps_train
for col in cc_apps_train.columns:
    # Check if the column is of object type
    if cc_apps_train[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps_train = cc_apps_train.fillna(cc_apps_train[col].value_counts().index[0])
        cc_apps_test = cc_apps_test.fillna(cc_apps_train[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
print(cc_apps_train.isnull().sum())
print(cc_apps_test.isnull().sum())



# ## 7. Preprocessing the data (part i)
# The missing values are now successfully handled.
# There is still some minor but essential data preprocessing needed before we proceed towards building our machine learning model. We are going to divide these remaining preprocessing steps into two main tasks:

# Convert the non-numeric data into numeric.
# Scale the feature values to a uniform range.

# First, we will be converting all the non-numeric values into numeric ones. We do this because not only it results
# in a faster computation but also many machine learning models (like XGBoost) (and especially the ones developed
# using scikit-learn) require the data to be in a strictly numeric format. We will do this by using the
# get_dummies() method from pandas.


# Convert the categorical features in the train and test sets independently
cc_apps_train = pd.get_dummies(cc_apps_train)
cc_apps_test = pd.get_dummies(cc_apps_test)

# Reindex the columns of the test set aligning with the train set
cc_apps_test = cc_apps_test.reindex(columns=cc_apps_train.columns, fill_value=0)

# ## 8. Preprocessing the data (part ii) Now, we are only left with one final preprocessing step of scaling before we
# can fit a machine learning model to the data.

# Now, let's try to understand what these scaled values mean in the
# real world. Let's use CreditScore as an example. The credit score of a person is their
# creditworthiness based on their credit history. The higher this number, the more financially trustworthy a person
# is considered to be. So, a CreditScore of 1 is the highest since we're rescaling all the values to the
# range of 0-1.

# In[502]:


# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Segregate features and labels into separate variables
X_train, y_train = cc_apps_train.iloc[:, :-1].values, cc_apps_train.iloc[:, [-1]].values
X_test, y_test = cc_apps_test.iloc[:, :-1].values, cc_apps_test.iloc[:, [-1]].values

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)


# ## 9. Fitting a logistic regression model to the train set

# Which model should we pick? A question to ask is: are the features that affect the credit card approval
# decision process correlated with each other? Although we can measure correlation, that is outside the scope of
# this notebook, so we'll rely on our intuition that they indeed are correlated for now. Because of this correlation,
# we'll take advantage of the fact that generalized linear models perform well in these cases. Let's start our
# machine learning modeling with a Logistic Regression model (a generalized linear model).

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train,y_train)


# ## 10. Making predictions and evaluating performance

# We will now evaluate our model on the test set with respect to classification
# accuracy. But we will also take a look the model's confusion matrix. In the case of
# predicting credit card applications, it is important to see if our machine learning model is equally capable of
# predicting approved and denied status, in line with the frequency of these labels in our original dataset. If our
# model is not performing well in this aspect, then it might end up approving the application that should have been
# approved. The confusion matrix helps us to view our model's performance from these aspects.


# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test,y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test,y_pred)


# ## 11. Grid searching and making the model perform better
# Our model was pretty good! In fact, it was able to yield an accuracy score of 100%.


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100,150,200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)



# ## 12. Finding the best performing model

# We have defined the grid of hyperparameter values and converted them into a single dictionary format which
# GridSearchCV() expects as one of its parameters. Now, we will begin the grid search to see which values perform
# best. We will instantiate GridSearchCV() with our earlier logreg model with all the data we have. We will also
# instruct GridSearchCV() to perform a cross-validation of five folds. We'll end the notebook by storing the
# best-achieved score and the respective best parameters. While building this credit card predictor, we tackled some
# of the most widely-known preprocessing steps such as scaling, label encoding, and missing value imputation. We
# finished with some machine learning to predict if a person's application for a credit card would get approved or
# not given some information about that person.

# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Fit grid_model to the data
grid_model_result = grid_model.fit(rescaledX_train, y_train)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))

# Extract the best model and evaluate it on the test set
best_model = grid_model_result.best_estimator_
print("Accuracy of logistic regression classifier: ", best_model.score(rescaledX_test,y_test))
