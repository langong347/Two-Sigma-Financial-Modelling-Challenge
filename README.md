# Two-Sigma-Financial-Modelling-Challenge
The data set contains anonymized features pertaining to a time-varying value for a financial instrument. Each instrument has an id. Time is represented by the 'timestamp' feature and the variable to predict is 'y' (likely asset returns). No further information is provided on the meaning of the features or the type of instruments that are included in the data. Number of features: 108; Length of data: about 5 years. We need to use the training data to predict the values of 'y'. https://www.kaggle.com/c/two-sigma-financial-modeling/data

There is abundance of collinearity and large amount of missing values (>20%) in the training data. To address each of the problems:

1) Since collinearity makes the features dependent on each other and the least square assumption, beta^T = (X^T X)^(-1)(X^T Y), will break breakdown and produce very large beta, our model will have very low tolerance of testing error (i.e., small noise in the testing variables can cause large prediction error). We can orthogonalize the features (e.g., principal component) or penalize large betas by regularization (e.g., ridge or lasso regression);

2) For missing values we may fill the data by sklearn's preprocessing functions such as MinMaxScaler or StandardScaler. Actually, in this problem MinMaxScaler gives better results perhaps due to the existence of large outliers.

Given the large amount of samples and relatively small number of features, I divided the data set into two parts: the model is trained on the first 60% of data (about 3 years) and is applied towards the last 40% at each timestamp, generating predictions of y real-time (as in actual trading/investment strategies). The model is scored by r^2 at each timestamp.

I tried both principal components and ridge regressions. Their results are summarized as follows:

1) 17 principal components explain 90% of variance while using all PCs worsens the prediction scores. Strong collearity may render the PCs with smaller variance unstable. 
2) Pincipal component method on average gives better prediction score than ridge regression. However, the best score of ridge regression is much higher than that of principal component.  
3) Under high penalty, ridge regression suppresses the coefficients of many features but two: 'technical_20' and 'technical_30'. This accidental finding is consistent with other users' observations, such as this analysis on the physical meaning of the two features: https://www.kaggle.com/chenjx1005/physical-meanings-of-technical-20-30

The results of PC and Ridge are published separately in "Results of Principal Component and Ridge Regression.docx'.

The Jupyter Notebook explores further on the relation of 'technical_20' and 'technical_30' to 'y'.
Path: Two-Sigma-Financial-Modelling-Challenge/run.ipynb
