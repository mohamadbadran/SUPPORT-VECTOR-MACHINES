# STEP #1 : # Import Libraries 

import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization

# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# STEP #2 : Transform all our data into a dataframe

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

# STEP #3: Visualizing the data

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
sns.countplot(df_cancer['target'], label = "Count") 
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 

# STEP #4: MODEL TRAINING

# Let's drop the target label coloumns
X = df_cancer.drop(['target'],axis=1)

# Recall our target class
y = df_cancer['target']

# Dividing our data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)

# Import the classifier ,classification_report and confusion_matrix
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)

# STEP #5: EVALUATING THE MODEL

y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict))

# STEP #6: IMPROVING THE MODEL - PART 1 (NORMALIZATION)

# Recall the minimum of the training data
min_train = X_train.min()
min_train

# Calculate the range of the training data
range_train = (X_train - min_train).max()
range_train

# Scaling the training data
X_train_scaled = (X_train - min_train)/range_train

# Our scale must be between ZERO and ONE 
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

# Recall the minimum of the testing data
# Calculate the range of the testing data
min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

# Fitting the classifier to the scaled training data
svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

# Predicting new results depending on the scaled training data
y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_test,y_predict))

# IMPROVING THE MODEL - PART 2 (C & γ parameters)

 # Creating grid for all possible values of C & γ
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
 
# Run the SVC on all C & γ parameters and get the best of them
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)

# Fit the grid to the scaled training data
grid.fit(X_train_scaled,y_train)

# Observe the best parameters
print(grid.best_params_)

# Predicting grid results 
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
print(classification_report(y_test,grid_predictions))

























