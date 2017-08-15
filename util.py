import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cluster_by_NaN(df):
    ids = sorted(df.id.unique())
    cols = df.columns.drop(['id','timestamp','y']).insert(0,'length')
    nan_df = pd.DataFrame(data=None,index=ids,columns=cols,dtype=float)
    for name,group in df.groupby('id'):
        for c in cols:
            if c == 'length':
                nan_df.loc[name,c] = len(group)
            else:
                nan_df.loc[name,c] = group[c].isnull().sum()
    capped = nan_df.copy()
    capped[capped > 100] = 100
    capped = capped.div(capped['length'],axis='index').drop(['length'],axis=1)
    _,labels = dbscan(capped.values)
    labels = pd.Series(labels,index=ids).to_dict()
    return capped,labels

def plot_df(df,xlabel,ylabel,title,display_all_xlabels=False):
    plt.clf()
    ax = 0
    ax = df.plot(lw=0.5)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if display_all_xlabels:
        ax.set_xticks(np.arange(df.shape[0]))
        ax.set_xticklabels(df.index.values,rotation=90,fontsize=5)
    return ax.get_figure()
