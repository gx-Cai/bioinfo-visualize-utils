## Author: CGX
## Time: 2022 01 29
## Description: This file is aim to plot the cross validation ROC curve and Calibration curve
## 2022.11.24 updated

from random import seed
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from numpy import interp
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

import pickle
from xgboost import XGBClassifier, XGBRegressor
from tqdm import trange,tqdm

def plot_roc_curve(
    X,y,
    classifier = svm.SVC(),
    seed=1,iter_times=10,
    ax=None,
    **figargs):

    if ax is None:
        fig, ax = plt.subplots()

    X = X.values
    y = y.values
    cv = KFold(n_splits=10,shuffle=True,random_state=seed)
    # n_samples, n_features = X.shape

    # # Add noisy features
    # random_state = np.random.RandomState(seed)
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # #############################################################################
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves


    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    f = plt.figure(**figargs)
    for _ in trange(iter_times):
        for train, test in cv.split(X, y):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, lw=1, alpha=0.3,
            #         #label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc)
            #         )

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='.55',
        # label='Chance',
        alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='#3a5bcc',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#7d83ca', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    return f

def plot_calibrate_curve(
    X, y, classifier,ax,n_iters=10,n_bins=10,**line_args):
    cv = KFold(n_splits=10,shuffle=True)
    all_data_y = []
    all_data_prob = []

    for _ in trange(n_iters):
        for train, test in cv.split(X, y):
            classifier.fit(X[train], y[train])
            y_pred = classifier.predict_proba(X[test])
            y_pred = y_pred[:, 1]
            all_data_y.extend(y[test])
            all_data_prob.extend(y_pred)

    # plot calibration curve for the model
    return CalibrationDisplay.from_predictions(
        y_true=all_data_y,
        y_prob=all_data_prob,ref_line=False,
        n_bins=n_bins,ax=ax,**line_args)


if __name__ == "__main__":
    
    ### A test example.

    df_exp, df_os, idmap,df_emt = ACGR_data().values()
    # df_exp = RPKM2TPM(df_exp)
    samples = df_exp.columns

    idmap = pd.DataFrame(idmap).T
    idmap = idmap[~idmap['ENTREZ_GENE_ID'].isna()]
    idmap = idmap['ENTREZ_GENE_ID'].apply(
        lambda x: x.replace(' ','').replace('//','/').replace('//','/').split('/')
        ).explode().astype(int)

    def get_acgr_X(geneset,return_type='max'):
        all_matched = idmap[idmap.isin(geneset)]
        # for same gene, calculate the mean
        if return_type=='mean':
            return df_exp.loc[all_matched.index,:].groupby(all_matched).mean()
        elif return_type=='max':
            return df_exp.loc[all_matched.index,:].groupby(all_matched).max()
        elif return_type=='raw':
            return df_exp.loc[all_matched.index,:]


    # ACGRidmap = pd.DataFrame(ACGRidmap).T

    # with open("../../models/ACGRMeta_Genes.pkl","rb") as f:
    #     genestr,genes,genesensg = pickle.load(f)

    def name2id(names,target = 'NCBI gene ID'):
        genemap = getGenemap()
        idx = genemap['Approved symbol'].isin(names)
        ret = genemap.loc[idx,target]
        ret.index = names
        return ret

    # metagenes = pd.read_csv("../../models/MetaRelatedGenes.csv",index_col=0)
    metagenes = pd.read_csv('../../models/coef_meta_os_protein.csv').iloc[:,0]
    metagenes_ncbi = name2id(metagenes).values

    classifier = XGBClassifier(
        min_child_weight=1,
        # subsample=0.8,
        # colsample_bytree=0.8,
        scale_pos_weight=1,
        eval_metric='auc',
    )
    
    ## 1. For Peri meta pred
    y = df_os['PeriMeta']
    X = get_acgr_X(metagenes_ncbi).T.loc[y.index,:]

    sns.set_style('white')
    f,ax = plt.subplots(1,1,dpi=300)

    plot_roc_curve(
        X,y,ax=ax,
        classifier=classifier,dpi=300,seed=1,iter_times=100)
    plot_calibrate_curve(
        X.values,y.values,classifier,n_iters=100,ax=ax,
        mew=2,mec='white',lw=2,c='#bf6e73',marker='o',markersize=8,
        label='Calibrate curve',
    )
    ax.set_title(
        "Performance of ACGR GC-PM predictor",
        fontdict={'fontweight':'semibold','fontsize':12}
        )
    ax.legend(edgecolor='white')
    ax.set_xlabel('False Positive Rate / Mean Predicted probability')
    ax.set_ylabel('True Positive Rate / Observed probability')

    f.savefig("../../figures/ACGR_GC-PM_Performance.pdf",bbox_inches='tight')

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    f,ax = plt.subplots(1,1,dpi=300)
    pca = TSNE()#PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ax.scatter(
        X_pca[:,0],X_pca[:,1],
        c=y,cmap='RdBu_r',alpha=0.5,
        linewidths=.5, 
        edgecolors='face',
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('TSNE-1')
    ax.set_ylabel('TSNE-2')
    cmap = plt.cm.RdBu_r
    ax.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker='o', color='w',
                label='GC-PM',
                markerfacecolor=cmap(0.99), 
                markersize=10),
            plt.Line2D(
                [0], [0], marker='o', color='w',
                label='Without GC-PM',
                markerfacecolor=cmap(0), 
                markersize=10)],
        edgecolor='white',
    )
    f.savefig("../../figures/ACGR_GC-PM_TSNE.pdf",bbox_inches='tight')
    
    ## 1. EMT subtype pred
    y = (df_os['FU']==3).astype(int)
    X = get_acgr_X(metagenes['NCBI gene ID']).T.loc[y.index,:]

    sns.set_style('white')
    f,ax = plt.subplots(1,1,dpi=300)

    plot_roc_curve(
        X,y,ax=ax,
        classifier=classifier,dpi=300,seed=1,iter_times=100)
    plot_calibrate_curve(
        X.values,y.values,classifier,n_iters=100,ax=ax,
        mew=2,mec='white',lw=2,c='#bf6e73',marker='o',markersize=8,
        label='Calibrate curve',
    )
    ax.set_title(
        "Performance of ACGR EMT subtype predictor",
        fontdict={'fontweight':'semibold','fontsize':12}
        )
    ax.legend(edgecolor='white')
    ax.set_xlabel('False Positive Rate / Mean Predicted probability')
    ax.set_ylabel('True Positive Rate / Observed probability')

    f.savefig("../../figures/ACGR_EMT_Performance.pdf",bbox_inches='tight')

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    f,ax = plt.subplots(1,1,dpi=300)
    pca = TSNE()#PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ax.scatter(
        X_pca[:,0],X_pca[:,1],
        c=y,cmap='RdBu_r',alpha=0.5,
        linewidths=.5, 
        edgecolors='face',
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('TSNE-1')
    ax.set_ylabel('TSNE-2')
    cmap = plt.cm.RdBu_r
    ax.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker='o', color='w',
                label='EMT subtype',
                markerfacecolor=cmap(0.99), 
                markersize=10),
            plt.Line2D(
                [0], [0], marker='o', color='w',
                label='others',
                markerfacecolor=cmap(0), 
                markersize=10)],
        edgecolor='white',
    )
    f.savefig("../../figures/ACGR_EMT_TSNE.pdf",bbox_inches='tight')
