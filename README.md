# SUPPORT-VECTOR-MACHINES
BREAST CANCER CLASSIFICATION USING SUPPORT VECTOR MACHINES

You can view the jupyter notebook from here https://nbviewer.jupyter.org/github/mohamadbadran/SUPPORT-VECTOR-MACHINES/blob/master/SVM%20-%20Cancer%20Classification-Copy1.ipynb

# PROBLEM STATEMENT

Predicting if the cancer diagnosis is benign or malignant based on several observations/features
30 features are used, examples:

  - radius (mean of distances from center to points on the perimeter)
  
  - texture (standard deviation of gray-scale values)
  
  - perimeter
  
  - area
  
  - smoothness (local variation in radius lengths)
  
  - compactness (perimeter^2 / area - 1.0)
  
  - concavity (severity of concave portions of the contour)
  
  - concave points (number of concave portions of the contour)
  
  - symmetry 
  
  - fractal dimension ("coastline approximation" - 1)
  
Datasets are linearly separable using all 30 input features

Number of Instances: 569

Class Distribution: 212 Malignant, 357 Benign

Target class:

   - Malignant
   
   - Benign
   
DATA SOURCE: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

In this code we will pass through these following steps : 

- Importing the libraries

- Importing cancer data from sklearn library

- Visualizing some data

- Model training

- Evaluating the model

- Improving the model - Part 1 (Normalization)

- Improving the model - Part 2 (C & gamma parameters)

 .Explanation : We get 34% accuracy before we improve our model, and 34% accuracy is bad, so we tried to improve the model first by Normalizaton (step - 1) we get better accuracy which is 96% , second we also tried to improve our model (Part - 2) by using C & gamma parameters and we get better accuracy which is 97% . In the first step of improving the model we tried to scale all our data between 0 and 1 and in the second step we get the best C & gamma parameters that gives us the highest accuracy.

















