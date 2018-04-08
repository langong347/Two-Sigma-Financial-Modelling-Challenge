import os,sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet,RidgeCV,LassoCV,\
        ElasticNetCV
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import IncrementalPCA as ipca
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import dbscan
import sklearn.metrics as metrics 
import util
import matplotlib.pyplot as plt
import time
import random
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,8)
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.grid'] = True

def data_clean(df,cols):
    '''
    df is a dataframe of stocks/timestamp by features
    '''
    for _id in df.id.unique():
        tmp = df[df.id==_id][cols]
        cond_na = (tmp.isnull().sum() == tmp.shape[0])
        # Dropping NaN columns
        cols_ = tmp.columns[~cond_na]
        cols_na = tmp.columns[cond_na]
        # Windsurfing
        for col in cols_:
            percentiles = tmp[col].quantile([0.05,0.95]).values
            tmp.loc[tmp[col] >= percentiles[1],col] = percentiles[1]
            tmp.loc[tmp[col] <= percentiles[0],col] = percentiles[0]
        # Normalizing
        scaler = StandardScaler()
        tmp[cols_] = scaler.fit_transform(tmp[cols_].fillna(tmp[cols_].mean()))
        tmp[cols_na] = tmp[cols_na].fillna(0.0)
        df.loc[df.id==_id,cols] = tmp

def plot_data(df,cols,cleaned=False):
    if cleaned:
        c = 'cleaned'
    else:
        c = 'raw'
    for _id in df.id.unique():
        plt.clf()
        ax = df[df.id==_id][cols].boxplot(rot=90)
        ax.get_figure().savefig('../Analysis/boxplot_%d_%s.png'%(_id,c))
    print("Boxplots done.")

class Model:
    def __init__(self,df,mode,pca):
        #self.scaler = MinMaxScaler() #StandardScaler()
        y = df.y.values
        x = df[cols].values
        #self.means = x.mean()
        #x = x.fillna(self.means)
        #x = x.values
        self.decomp = False
        self.model = None

        #x = self.scaler.fit_transform(x)
        if pca:
            var_exp_tol = 0.9
            self.decomp = ipca()
            x_mod = self.decomp.fit_transform(x)
            cum_var_exp = self.decomp.explained_variance_ratio_.cumsum()
            self.n = np.arange(0,x.shape[1])[cum_var_exp >= var_exp_tol][0]
            self.n = min(self.n,int(np.sqrt(df.shape[0])))
            x = x_mod[:,0:self.n]
            print("# PC used:",self.n,"variance explained:",var_exp_tol)
        if mode == 'A':
            self.model = 'linreg'
            self.regr = LinearRegression()
        elif mode == 'B':
            self.model = 'ridge'
            #self.regr = Ridge(alpha = 200,random_state = 52)
            self.regr = RidgeCV(alphas=np.arange(0.1,1.1,0.1)*1E5,\
                        store_cv_values=True)
                        #scoring='neg_mean_squared_error')
        elif mode == 'C':
            self.model = 'lasso'
            #self.regr = Lasso(alpha = 1, selection = 'random', random_state = 52)
            self.regr = LassoCV(alphas=np.array([0.1,1,10]) * 1E4)
        elif mode == 'D':
            self.model = 'elasticNet'
            #self.regr = ElasticNet(alpha = 1,l1_ratio = 0.2, selection = 'random', random_state = 52)
            self.regr = ElasticNetCV(l1_ratio=np.arange(0.1,1.1)*0.1,\
                        n_alphas=3,alphas = np.array([0.01,0.1,1])*1E7)
        elif mode == 'E':
            self.model = 'ETR'
            self.regr = ETR(n_estimators=500,max_depth=10,\
                            max_features='auto',\
                            bootstrap=True,\
                            criterion='mse',\
                            #verbose=1,\
                            oob_score=True,\
                            n_jobs=4,random_state=50)
            #est = ETR(random_state=50,verbose=1,n_jobs=-1)
            tuned_parameters = [{'n_estimators': [10,100],
                                 'max_depth': [5,50]
                                }]
            #self.regr = GridSearchCV(estimator=est,\
            #                        param_grid=tuned_parameters,\
            #                        verbose=1)
        self.regr.fit(x,y)
      

    def predict(self,x):
        #x = x.fillna(self.means)
        x = x.values
        #x = self.scaler.transform(x)
        if self.decomp:
            x = self.decomp.transform(x)[:,0:self.n]
        return self.regr.predict(x)

    def __str__(self):
        return self.model

def score(y_true,y_pred):
    return {'r2':metrics.r2_score(y_true,y_pred),
            'mse':metrics.mean_squared_error(y_true,y_pred)}

def adjust():
    filename = '../Data/train_clean.h5'
    df = pd.HDFStore(filename).get("train")
    #df1 = pd.HDFStore('../Analysis/pred_linreg.h5').get('pred')
    df1 = pd.HDFStore('../Analysis/pred_ridge.h5').get('pred')
    df2 = pd.HDFStore('../Analysis/pred_ETR.h5').get('pred')
    print("All files loaded.")
    #-----additional adjustment---------------
    start = time.time()
    print("Additional adjustment starts at %s"%(time.strftime('%H:%M:%S', \
            time.localtime(start))))
    y_median = df.groupby('id').y.median()
    y_max = df.y.max()
    y_min = df.y.min()
    y_med = df1.apply(lambda row:y_median[row.id]\
                            if row.id in y_median else 0,axis=1)
    #y_med = 0
    y_pred1 = (df1.y_pred+y_med).clip(y_min,y_max)
    y_pred2 = (df2.y_pred).clip(y_min,y_max)
    print("Additional adjustment took %.2f seconds." %(time.time() - start))
    y_true = df1.y
    return y_true,y_pred1,y_pred2

def mix(y_pred1,y_pred2,w=0.3):
    y_pred = y_pred1 * w + y_pred2 * (1-w)
    return y_pred

#----------read data------------------------
filename = '../Data/train.h5'
df = pd.HDFStore(filename).get("train")
print("%s loaded."%filename) 
print("Shape of data: (%d,%d)"%df.shape)
cols = df.columns.drop(['y','timestamp','id']).tolist()
train = df.timestamp.unique()[0:1200]
test = df.timestamp.unique()[1200:]
#----------data exploration-------------------
'''
from scipy.stats import signaltonoise
df[cols] = df[cols].fillna(df[cols].mean())
sn = signaltonoise(df[['y'] + cols])
'''
#ids = random.sample(df.id.unique().tolist(),5)
#cols = random.sample(cols,20)

# select ids that are longer than 1200 timestamps
#id_len = df.groupby('id')['id'].count()
#ids = random.sample(id_len[id_len >= 1200].index.values.tolist(),5)
'''
# construct new features
new_feature = 'tech2030'
df[new_feature] = df.technical_20 - df.technical_30

# time-wise stat of df
grp = df.groupby('timestamp')[cols + [new_feature] + ['y']]
stat = grp.agg(['mean','std'])
stat = stat.swaplevel(axis=1)
'''
#plot rolling mean of y for different time segements
'''
for w in [1,5,10]:
    tmp = stat['mean']['y'].rolling(window=w,win_type='triang').mean()
    tmp.name = 'y_roll_mean'
    #tmp = stat['mean']['y']
    tmp2 = pd.concat([tmp,stat['mean'][new_feature]],axis=1)
    t0 = 0
    for t in [658,905,1248,1812]:
        plt.clf()
        ax = tmp2.loc[t0:t].plot(secondary_y=tmp.name,title='window=%d'%w)
        #ax.set_ylabel('mean of %s'%new_feature)
        #ax2 = ax.twinx()
        #ax2.set_ylabel('%d day rolling mean of y'%w)
        ax.get_figure().savefig('../Analysis/yroll_tech2030_%d_%d.png'%(w,t))
        t0 = t+1
'''
'''
# plot mean and std of y and constructed features
plt.clf()
ax = stat['mean'][['y',new_feature]].plot(secondary_y=['y'])
ax.get_figure().savefig('../Analysis/mean_y_%s.png'%new_feature)
plt.clf()
ax = stat['std'][['y',new_feature]].plot(secondary_y=['y'])
ax.get_figure().savefig('../Analysis/std_y_%s.png'%new_feature)
'''
"""
for id_ in ids:
    tmp = df[df.id==id_]
    tmp = tmp.set_index(tmp.timestamp)
    '''
    # plot time series
    plt.clf()
    ax = tmp.loc[0:1200,new_feature].plot()
    ax.set_xlabel('timestamp')
    ax.get_figure().savefig('../Analysis/tsplot_%s_id%d.png'%\
                            (new_feature,id_))
    '''
    '''
    # plot features
    plt.clf()
    ax = tmp[cols].plot(kind='box',grid=True,rot=90,vert=True)
    ax.set_ylabel('Feature Values')
    ax.get_figure().savefig('../Analysis/boxplot_features_id%d.png'%id_)
    plt.clf()
    nan_stat = tmp[cols].isnull().sum()/tmp.shape[0]
    ax = nan_stat.plot(kind='bar',grid=True)
    ax.set_ylabel('Fraction of NaNs')
    ax.get_figure().savefig('../Analysis/barplot_NaNs_id%d.png'%id_)
    '''    
    '''
    # plot time series plot of y
    plt.clf()
    ax = tmp.y.plot(grid=True,rot=90,figsize=(12,8),fontsize=8)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.get_figure().savefig('../Presentation/tsplot_y_id%d.png'%id_)
    '''
"""
'''
# plot y distribution for all ids
tmp = df.groupby('id').y.mean().sort_values()
y_mean = df.groupby('timestamp').y.mean()
ids = tmp.index.values[np.arange(0,tmp.shape[0],50)]
plt.clf()
tmp = df[df.id.isin(ids)].pivot(index='timestamp',columns='id',values='y')
#tmp['<y>_t'] = y_mean
ax = tmp.plot(kind='box',figsize=(12,8),rot=90,grid=True)
ax.set_xlabel('id')
ax.set_ylabel('y')
ax.get_figure().savefig('../Presentation/boxplot_y_by_id.png')
'''

#----------data cleaning---------------------
##Select a few ids to test first
#rand_ids = random.sample(df.id.unique().tolist(),100)
#df = df[df.id.isin(rand_ids)]
#plot_data(df,cols)
'''
start = time.time()
print("Data cleaning starts at %s"%(time.strftime('%H:%M:%S', \
        time.localtime(start))))
data_clean(df,cols)
print("Data cleaning done: %.2f seconds. "% (time() - start))
#plot_data(df,cols,cleaned=True)
clean_data_file = '../Data/train_clean.h5'
if os.path.exists(clean_data_file):
    os.remove(clean_data_file)
store = pd.HDFStore(clean_data_file)
store['train'] = df
print("Clean data saved.")
'''
#df[cols] = df[cols].fillna(df[cols].mean())
#scalar = StandardScaler()
#df[cols] = scalar.fit_transform(df[cols])
#--------delete boundary y points on the two ends-----------------------
'''
max_y = df.y.max().max()
min_y = df.y.min().min()
df = df[~((df.y==max_y)|(df.y==min_y))]
'''
#---------add lag features---------------------------------
'''
#ids = random.sample(df.id.unique().tolist(),10) #Debug
#ids = df.id.unique()
y_mean = df.groupby('timestamp').y.mean()
start = time()
lags_ = [5]#[1,5,10,30,90]
for lag in lags_:
    df['lag_%d'%lag] = pd.Series(None)
    lag_y = y_mean.shift(lag).fillna(0)
    for t in df.timestamp.unique():
        df.loc[df.timestamp==t,'lag_%d'%lag] = lag_y[t]
    cols = cols + ['lag_%d'%lag]
#for id_ in ids:
#    print("Warning: adding lags by id will take a long time to finish!")
#    tmp = df.loc[df.id==id_]
#    for lag in lags_:
#        tmp.loc[:,'lag_%d'%lag] = tmp['y'].shift(lag).fillna(0)
#    df.loc[df.id==id_] = tmp
print("Lag features added: %.2f seconds. "% (time() - start))
'''
#-----------------------add variance features--------------------
'''
start = time.time()
df['var'] = pd.Series(None)
for t in df.timestamp.unique():
    var1 = df[df.timestamp==t]['technical_20'].var()
    #df.loc[df.timestamp==t,'var_2']=df[df.timestamp==t]['technical_30'].var()
cols = cols + ['var_1','var_2']
print("Variance features added: %.2f seconds. "% (time() - start))
'''
#----statistics of number of NaN values of coefficients per id------------------
'''
from sklearn.cluster import DBSCAN
id_len = df.groupby('id')['timestamp'].count()
id_len.name = 'len'
id_stat_na = df.groupby('id')[cols].apply(lambda x:x.isnull().sum())
id_stat = pd.concat([id_len,id_stat_na],axis=1)
#pct_na = id_stat.apply(lambda x:x[cols]/float(x['len']),axis=1)
pct_na = id_stat.div(id_stat['len'],axis='index').drop(['len'],axis=1)
pct_na[pct_na > 0.2] = 1
db_train = pct_na
db = DBSCAN(min_samples=20)
db.fit(db_train[cols])
db_train['cluster'] = db.labels_
db_train['id'] = db_train.index.values
plt.clf()
ax = db_train.groupby('cluster')['id'].count().plot(kind='bar',grid=True,\
        xlim=(-0.5,5))
ax.get_figure().savefig('../Analysis/cluster_ids_by_nan.png')
c_stat = db_train.groupby('cluster')['id'].count()
print('eps=%s'%db.eps)
print('min_samples=%d'%db.min_samples)
print(c_stat)
'''
#-----------auto correlation analyses------------------------
'''
from pandas.tools.plotting import autocorrelation_plot
rand_ids = random.sample(df.id.unique().tolist(),100)
for id_ in rand_ids:
    ax = autocorrelation_plot(df[df.id==id_].y)
    ax.get_figure().savefig('../Analysis/autocorr_y_id_%d.png'%id_)
    plt.clf()
'''
#-----construct tech20-tech30 feature------------------
'''
df['tech2030'] = df['technical_20'] - df['technical_30']
cols = ['tech2030']
'''
#----training--------------------------------------------------------
'''
#debug
#print("Warning: debug mode on!")
#ids = random.sample(df.id.unique().tolist(),10)
#df = df[df.id.isin(ids)]
#print(df.shape)

start = time.time()
print("Training starts at %s"%(time.strftime('%H:%M:%S', \
        time.localtime(start))))
model = Model(df[df.timestamp.isin(train)][cols + ['y']],mode='B',pca=False)
print("Regressor %s took %.2f seconds for training." %\
        (model,(time.time() - start)))

try:
    cv_results = pd.DataFrame(model.regr.cv_results_).T
    cv_results.to_csv('../Analysis/cv_results_%s.txt'%model)
except:
    print("No CV results to be stored.")

df['y_pred'] = pd.Series(data=None)
'''
#-----prediction by timestamp (to save memory,e.g., random forest--------
'''
for t in test:
    if t % 100 == 0:
        print("Timestamp %d"%t)
    test_data = df[df.timestamp == t][cols]
    y_pred = model.predict(test_data)
    df.loc[df.timestamp == t,'y_pred'] = y_pred

#-----------------------------------------------

y_true = df[df.timestamp.isin(test)].y
y_pred = df[df.timestamp.isin(test)].y_pred
scores = score(y_true,y_pred)
pd.Series(scores).to_csv('../Analysis/score_%s.txt'%model)
pred_data=df.loc[df.timestamp.isin(test),['id','timestamp','y','y_pred']]
pred_data_file = '../Analysis/pred_%s.h5'%model
if os.path.exists(pred_data_file):
    os.remove(pred_data_file)
store = pd.HDFStore(pred_data_file)
store['pred'] = pred_data
print("Prediction results saved.")
'''
#-----training by id------------------------------------
'''
id_cnt = df.groupby('id')['id'].count()
id_cnt.name = 'count'
data_train = df[df.timestamp.isin(train)]
data_test = df[df.timestamp.isin(test)]
data_test['y_pred'] = pd.Series(data=None)
start = time()
model0 = Model(data_train[cols+['y']],mode='A',pca=True)
models = {}
for id_ in data_train.id.unique():
    if id_cnt[id_] < len(train) * 0.6: continue
    model = Model(data_train[data_train.id==id_][cols+['y']],mode='A',pca=True)
    models[id_] = model
print("Regressor took %.2f seconds for training." %(time() - start))

for id_ in data_test.id.unique():
    tmp = data_test[data_test.id==id_][cols]
    if id_ in models:
        y_pred = models[id_].predict(tmp)
    else:
        y_pred = model0.predict(tmp)
    data_test.loc[df.id==id_,'y_pred'] = y_pred

y_true = data_test.y
y_pred = data_test.y_pred
scores = score(y_true,y_pred)
pd.Series(scores).to_csv('../Analysis/score_train_by_id_%s.txt'%model)
'''
#-----------analyses---------------------------------------
'''
# Plot quantiles of y_true,y_pred for the test data
plt.clf()
df1 = df.groupby('timestamp')[['y','y_pred']].quantile([0.05,0.5,0.95]).\
        unstack()
ax = df1[df1.index.isin(test)].plot(figsize=(30,15)) 
ax.set_ylabel('percentiles')
ax.set_xlabel('timestamp')
ax.get_figure().savefig('../Analysis/quantiles_%s.png'%model)


# Plot feature importance graph
if hasattr(model.regr,'feature_importances_'):
    importance = model.regr.feature_importances_
    importance = pd.DataFrame(importance, index=cols,columns=["Importance"])
    importance["Std"] = np.std([tree.feature_importances_
                                for tree in model.regr.estimators_], axis=0)
    importance = importance.sort_values(by="Importance")
y = importance.ix[:, 0]
yerr = importance.ix[:, 1]
plt.clf()
fig,ax = plt.subplots()
y.plot.bar(yerr=yerr,ax=ax,grid=True)
importance.to_csv('../Analysis/feature_importance.txt')
ax.set_ylabel('feature importance')
fig.savefig('../Analysis/feature_importance_%s.png'%model)
'''
'''
# Plot coefficients
df2 = pd.Series(model.regr.coef_,index=cols).sort_values()
df2.to_csv('../Analysis/coef_%s.txt'%model)
#df2 = pd.concat([df2.head(),df2.tail()])
plt.clf()
ax=df2.plot(kind='bar',grid=True)
ax.set_ylabel('coef')
ax.get_figure().savefig('../Analysis/coef_%s.png'%model)
'''
'''
# CV ERROR
cv_error = pd.DataFrame(model.regr.cv_values_,columns=model.regr.alphas)
cv_error.to_csv('../Analysis/cv_error_%s.txt'%model)
plt.clf()
ax = (cv_error.mean()-cv_error.mean().mean()).\
        plot(grid=True,figsize=(12,8))
ax.set_ylabel('Cross Validation Error')
ax.set_xlabel('Parameter')
ax.get_figure().savefig('../Analysis/cv_error_%s.png'%model)
'''
