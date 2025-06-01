from scipy.stats import pearsonr, spearmanr
from scipy.stats import kruskal, mannwhitneyu, fisher_exact
import itertools
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os

def regression_scatter_line(
    mdf, regression_x, regression_y, hue = None, 
    corr_type = 'pearson',cmap = 'Set2', ax=None, 
    show_plots = True,
    **legend_kwargs
):
    from scipy.stats import pearsonr, spearmanr
    def scatter_reg_lines(ax, label = regression_y):
        sns.set_style('whitegrid')
        rho, pval = corr_func(mdf[label],mdf[regression_x])

        if hue is None:
            ax.scatter(x = mdf[label], y = mdf[regression_x], color = platte(0.1))
        else:
            for i,cat in enumerate(sorted(mdf[hue].unique())):
                subdf = mdf.query(f'{hue} == @cat')
                ax.scatter(
                    x = subdf[label],
                    y = subdf[regression_x],
                    color = platte(i),
                    label = cat
                )

        regx = mdf.loc[:,label]
        regy = mdf.loc[:,regression_x]
        z = np.polyfit(regx, regy, 1)
        p = np.poly1d(z)
        x= [regx.min(), regx.max()]
        y= [p(regx.min()), p(regx.max())]
        ax.plot(x,y)
        ax.scatter(x=regx.mean(),y=regy.mean(), s = 0, label = f'rho = {rho:.2f}')
        ax.scatter(x=regx.mean(),y=regy.mean(), s = 0, label = f'p = {pval:.2f}')
        default_legend_kwargs = dict(frameon=False, bbox_to_anchor = (0.95,1.15), fontsize = 10)
        default_legend_kwargs.update(legend_kwargs)
        ax.legend(**default_legend_kwargs)
        ax.set_xlabel(label)
        ax.set_ylabel(f"ratio of {regression_x}")
        return rho, pval
    
    if ax is None:
        ax = plt.gca()

    platte = plt.get_cmap(cmap)
    corr_func = {'pearson':pearsonr,'spearman':spearmanr}[corr_type]
    mdf = mdf[[regression_y,regression_x,hue]].dropna() if hue is not None else mdf[[regression_y,regression_x]].dropna()
    if corr_type == 'spearman':
        mdf.loc[:,[regression_y,regression_x]] = mdf.loc[:,[regression_y,regression_x]].rank(axis=0, method='average')
    
    rho, pval = scatter_reg_lines(ax,regression_y)
    stats = {'rho':rho,'pval':pval}
    if not show_plots: plt.close()
    
    return {
        'ax': ax,
        'stats': stats
    }





def ratio_sig_test(dfvis,value_col,groupby,stat_method = "H-test",**kwargs):
    from . import box_with_scatter
    def fisher_continous_test(data1,data2):
        data1,data2 = np.array(data1), np.array(data2)
        mid = np.median(np.concatenate([data1,data2]))
        contingency_table = [
            [(data1 > mid).sum(),(data1 <= mid).sum()],
            [(data2 > mid).sum(),(data2 <= mid).sum()]
        ]
        return fisher_exact(contingency_table)

    stat_func = {
        'H-test': kruskal,
        'U-test': mannwhitneyu,
        'Fisher': fisher_continous_test
    }[stat_method]

    result = {}
    for j in dfvis[groupby].unique():
        if (stat_method != 'H-test'): continue
        try: _, p = stat_func(*dfvis.groupby([j])[value_col].apply(lambda x: x.values).values.tolist())
        except Exception as e: p = 1; print(e)
        result[j] = p

    box_with_scatter(
        dfvis, x = groupby, y = value_col,
        labeling=[
            f"{j} p = {p:.2f}"
            for j,p in result.items()
        ]
        **kwargs
    )
    
    return result