import os,sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet,RidgeCV,LassoCV
from sklearn.decomposition import IncrementalPCA as ipca
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import dbscan
from sklearn.metrics import r2_score
import util

class Model:
    def __init__(self,df,mode,pca):
        self.scaler = MinMaxScaler() #StandardScaler()
        self.cols = df.columns.drop(['y','timestamp','id'])
        self.cols = ['technical_20','technical_30']
        y = df.y.values
        x = df[self.cols]
        self.means = x.mean()
        x = x.fillna(self.means)
        x = x.values
        self.decomp = False

        x = self.scaler.fit_transform(x)
        if pca:
            var_exp_tol = 0.9
            self.decomp = ipca()
            x_mod = self.decomp.fit_transform(x)
            cum_var_exp = self.decomp.explained_variance_ratio_.cumsum()
            self.n = np.arange(0,x.shape[1])[cum_var_exp >= var_exp_tol][0]
            x = x_mod[:,0:self.n]
            print("# PC used:",self.n,"variance explained:",var_exp_tol)
        if mode == 'A':
            self.regr = LinearRegression()
        elif mode == 'B':
            self.regr = Ridge(alpha = 200,random_state = 52)
            #self.regr = RidgeCV(alphas=(1,10,100,200),store_cv_values=True)
        elif mode == 'C':
            #self.regr = Lasso(alpha = 1, selection = 'random', random_state = 52)
            self.regr = LassoCV(eps=0.001,n_alphas=100,random_state=None,selection='cyclic')
        elif mode == 'D':
            self.regr = ElasticNet(alpha = 1,l1_ratio = 0.2, selection = 'random', random_state = 52)
        self.regr.fit(x,y)
      

    def predict(self,x):
        x = x[self.cols]
        x = x.fillna(self.means)
        x = x.values
        x = self.scaler.transform(x)
        if self.decomp:
            x = self.decomp.transform(x)[:,0:self.n]
        return self.regr.predict(x)

    def score(self,y_true,y_pred):
        return r2_score(y_true,y_pred)



#----------------------------------------

df = pd.HDFStore('../Data/train.h5','r').get("train")
train = df.timestamp.unique()[0:906]
test = df.timestamp.unique()[906:]
model = Model(df[df.timestamp.isin(train)],mode='A',pca=False)
print(model.cols[np.arange(model.regr.coef_.shape[0])[model.regr.coef_ == model.regr.coef_.min()][0]])
print(model.cols[np.arange(model.regr.coef_.shape[0])[model.regr.coef_ == model.regr.coef_.max()][0]])

score = pd.Series(data=None,index=test,name='score')
df['y pred'] = pd.Series(data=None)
for t in test:
    if t % 100 == 0:
        print("Timestamp %d"%t)
    df_t = df[df.timestamp == t]
    y_true = df_t.y
    y_pred = model.predict(df_t)
    score[t] = model.score(y_true,y_pred)
    df.loc[df.timestamp == t,'y pred'] = y_pred
score.index.name = 't'


