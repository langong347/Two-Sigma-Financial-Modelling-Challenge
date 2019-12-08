# Two-Sigma-Financial-Modelling-Challenge
The data set contains anonymized features pertaining to a time-varying value for a financial instrument. Each instrument has an id. Time is represented by the 'timestamp' feature and the variable to predict is 'y' (presumably asset returns). No further information is provided on the meaning of the features or the type of instruments that are included in the data. Number of features: 108; Length of data: about 5 years. We need to use the training data to predict the values of 'y'. https://www.kaggle.com/c/two-sigma-financial-modeling/data

There is abundance of collinearity and large amount of missing values (>20%) in the training data.

1) Since collinearity makes the features dependent on each other and the least square assumption, beta^T = (X^T X)^(-1)(X^T Y), will break breakdown and produce very large beta, our model will have very low tolerance of testing error (i.e., small noise in the testing variables can cause large prediction error). We can orthogonalize the features (e.g., principal component) or penalize large betas by regularization (e.g., ridge or lasso regression);

2) Data preprocessing is done for each instrument via: clipping the outliers, standardizing feature values and NaN filling.

3) 'y' has very low signal-to-noise ratio, suggesting the challenge of attaining a high prediction accuracy.

I divided the data set into two parts: the model was trained on the first 60% of data (about 3 years) and applied towards the last 40%. I used r^2 score to evaluate accuracy of prediction.

Modeling Approaches and Results:

1) I have tried Principal components, Ridge Regression and Random Forest. However, the greatest improvement came from using mixed approaches (by 111% from simple linear regression on principal components).

2) Two features 'technical_20' and 'technical_30' are shown to be the most important. Further analysis on these two features suggests their difference might be related to short-term rolling mean of 'y', and their variance is almost in sync with 'y' as well. In finance, this may correspond to portfolio strategies such as having the asset return ('y') track general market indices (presumably 'technical_20' and 'technical_30').

# Files

[A good overview of my results and analysis](ResultsSummary_TwoSigmaChallenge.pdf) 

[Python code](run.v5.py)

[Further analysis of features technical\_20 and technical\_30 to y](Further Analysis.ipynb)

