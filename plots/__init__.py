import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import os
import seaborn as sns

def Enrichr_bar_group(
    data,
    color_col = 'Adjusted P-value',
    x_col = "Odds Ratio",
    group_col = "Gene_set",
    name_col = "Term",
    show_n = 5, 
    cmap=cm.Set1,
    colorbar_cmap=cm.coolwarm, 
    width_height_ratio=2, 
    **figargs
    ):
    
    data = data.groupby(group_col).apply(lambda x: x.sort_values(by=color_col).iloc[0:show_n,:])
    lable_max = int(np.log10(data[color_col].min()))
    fig = plt.figure(constrained_layout=True, **figargs)
    gs = fig.add_gridspec(
        data.shape[0], data.shape[0]*width_height_ratio)
    used_gride = 0
    xlim_max = int(data[x_col].max()*10+1)/10
    xlim_min = int(data[x_col].min()*10-1)/10

    axes = []
    for n, cat in enumerate(data[group_col].unique()):
        ctarget = data[data[group_col] == cat]
        ax = fig.add_subplot(gs[used_gride:used_gride+ctarget.shape[0], :])
        used_gride += ctarget.shape[0]
        ax.barh(
            width=ctarget[x_col],
            y=range(ctarget.shape[0]),
            color=[colorbar_cmap(np.log10(i)/lable_max) for i in ctarget[color_col]])
        ax.set_yticks([])
        for i,term in enumerate(ctarget[name_col]):
            ax.text(xlim_max / 50 + xlim_min, i, term, verticalalignment='center', fontsize=8,color='.25')
        # ax.set_yticklabels(ctarget.index.values.tolist())
        ax.tick_params("y", labelcolor=cmap(n))
        ax.set_title(cat)
        # ax.text(
        #     1.02, 0, cat,
        #     transform=ax.transAxes, rotation=90, color=cmap(n), verticalalignment='bottom',
        #     weight='semibold')
        axes.append(ax)

    ## setting all axes in the figure share the x axis
    for ax in axes:
        ax.set_xlim(xlim_min, xlim_max)

    for ax in axes[:-1]:
        ax.set_xticklabels([])
        ax.set_xticks([])
    plt.xlabel(x_col)

    cbar_ax = fig.add_axes([1.01, 0.15, 0.03, 0.3])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=colorbar_cmap), cax=cbar_ax)
    cbar.set_ticks(ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels([0, -lable_max/4, -lable_max /
                         2, -3*lable_max/4, -lable_max])
    cbar.set_label(f"log10 ({color_col})")

    return fig

def Enrichr_bar_group2(
    data, color_col, x_col, group_col, name_col = "Term",
    show_n = 5, 
    cmap=cm.Set1,
    colorbar_cmap=cm.coolwarm, 
    width_height_ratio=2,
    cbar_ax_loc = [0.085, 1.02, 0.91, 0.02],
    lable_max = None,
    **figargs
    ):
    
    data = data.groupby(group_col).apply(lambda x: x.sort_values(by=color_col).iloc[0:show_n,:])
    lable_max = int(data[color_col].max()) if lable_max is None else lable_max
    fig = plt.figure(constrained_layout=True, **figargs)
    gs = fig.add_gridspec(
        data.shape[0], data.shape[0]*width_height_ratio)
    used_gride = 0
    xlim_max = int(data[x_col].max()*10+1)/10
    xlim_min = int(data[x_col].min()*10-1)/10

    axes = []
    for n, cat in enumerate(data[group_col].unique()):
        ctarget = data[data[group_col] == cat]
        ax = fig.add_subplot(gs[used_gride:used_gride+ctarget.shape[0], :])
        used_gride += ctarget.shape[0]
        bar_colors = [colorbar_cmap(i/lable_max) for i in ctarget[color_col]]
        ax.barh(
            width=ctarget[x_col],
            y=range(ctarget.shape[0]),
            color=bar_colors)
        ax.set_yticks([])
        for i,term in enumerate(ctarget[name_col]):
            r,g,b,a = bar_colors[i]
            ax.text(
                xlim_max/50*49, i, term,
                verticalalignment='center', fontsize=8,color='.2',
                horizontalalignment ='right'
                # path_effects=[pe.Stroke(linewidth=.75, foreground='k'), pe.Normal()]
            )
        # ax.set_yticklabels(ctarget.index.values.tolist())
        ax.tick_params("y", labelcolor=cmap(n))
        # ax.set_title(cat)
        # ax.text(
        #     1.02, 0, cat,
        #     transform=ax.transAxes, rotation=90, color=cmap(n), verticalalignment='bottom',
        #     weight='semibold')
        ax.set_ylabel(cat)
        axes.append(ax)

    ## setting all axes in the figure share the x axis
    for ax in axes:
        ax.set_xlim(xlim_min, xlim_max)

    for ax in axes[:-1]:
        ax.set_xticklabels([])
        ax.set_xticks([])
    plt.xlabel(x_col)

    # cbar_ax = fig.add_axes([1.01, 0.07, 0.03, 0.3])
    cbar_ax = fig.add_axes(cbar_ax_loc)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=colorbar_cmap), cax=cbar_ax, location ='top')
    cbar.set_ticks(ticks=[0, 1])
    cbar.set_ticklabels([0, lable_max])
    cbar.set_label(f"{color_col}")
    
    return fig

def plot_volcano(
    df,x,y,xthres,ythres,
    label_items:list=[],
    ax=None, label_offset=0.1,
    cmap = 'RdBu_r',
    label_not_sig=False,
    line_kwargs = {},
    x_compare_abs=True,
    y_compare_abs=False,
    **scater_kwargs
):
    from adjustText import adjust_text

    default_scater_kwargs = {
        's': 10,
        'alpha': 0.5,
    }
    default_scater_kwargs.update(scater_kwargs)
    if type(cmap) is str:
        marker_color_red = plt.get_cmap(cmap)(0.8)
        marker_color_blue = plt.get_cmap(cmap)(0.2)
        font_color_red = plt.get_cmap(cmap)(0.9)
        font_color_blue = plt.get_cmap(cmap)(0.1)
    else:
        marker_color_red = cmap[0]
        marker_color_blue = cmap[1]
        font_color_red = cmap[2]
        font_color_blue = cmap[3]

    if ax is None:
        ax = plt.gca()
    if x_compare_abs:
        df["abs_x"] = np.abs(df[x])
        notsig_idx = df["abs_x"] < xthres
    else:
        notsig_idx = df[x] < xthres
    if y_compare_abs:
        df["abs_y"] = np.abs(df[y])
        notsig_idy = df["abs_y"] < ythres
    else:
        notsig_idy = df[y] < ythres

    not_sig = notsig_idx | notsig_idy
    up_idx = df[x] >= 0
    down_idx = df[x] < 0

    ax.scatter(
        df.loc[not_sig,x],
        df.loc[not_sig,y],
        color='.8',
        **default_scater_kwargs
    )
    ax.scatter(
        df.loc[~not_sig & up_idx,x],
        df.loc[~not_sig & up_idx,y],
        color=marker_color_red,
        **default_scater_kwargs
    )
    ax.scatter(
        df.loc[~not_sig & down_idx,x],
        df.loc[~not_sig & down_idx,y],
        color=marker_color_blue,
        **default_scater_kwargs
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    default_line_kwargs = dict(
        color='.8',linestyle='--'
    )
    default_line_kwargs.update(line_kwargs)

    ax.vlines(
        x=xthres if not x_compare_abs else [xthres, -xthres],
        ymin=0,ymax = ax.get_ylim()[1],
        **default_line_kwargs
    )

    ax.hlines(
        y=ythres,
        xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1],
        **default_line_kwargs
    )
    if len(label_items) != 0:
        texts = []
        for gene in label_items:
            if gene in df.index:
                if not_sig.loc[gene] and not label_not_sig: continue 
                font_color = [font_color_red,font_color_blue][bool(down_idx.loc[gene])] if not not_sig.loc[gene] else '.8'
                texts.append(
                    ax.text(
                        df.loc[gene,x]+label_offset,
                        df.loc[gene,y]+label_offset,
                        gene,
                        fontsize=10,
                        color= font_color
                ))
        if len(texts) == 0:
            raise ValueError('No label items in the dataframe, please check the `label_items` list')
        adjust_text(texts,ax=ax,arrowprops=dict(arrowstyle='-',color='grey',lw=0.5))
    
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Not significant', markerfacecolor='.8', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Up-regulated', markerfacecolor=marker_color_red, markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Down-regulated', markerfacecolor=marker_color_blue, markersize=10),
        ],
        ncol=3,
        bbox_to_anchor=(0.5, 1.1), frameon=False,
        loc='upper center', 
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    sns.despine(
        top=True, right=True, left=False, bottom=False, 
        offset= 10, trim=False
    )
    
    return ax

def box_with_scatter(
        data, x, y, hue=None,
        ax = None, palette='Set2', labeling:list=[],
        stripplot_kwargs = dict(), boxplot_kwargs = dict()):
    # TODO: hue of stripplot support
    # TODO: subsample support for large data of less point in stripplot
    sns.set_palette(palette)
    if ax is None:
        ax = plt.gca()

    default_boxplot_kwargs = dict(
        fliersize=0
    )
    default_boxplot_kwargs.update(boxplot_kwargs)
    sns.boxplot(
        x=x, y=y, hue=hue,
        data=data, 
        ax=ax,
        **default_boxplot_kwargs
        )
    
    # subsample
    if data.shape[0] > 1000:
        data = data.sample(1000)
    
    default_stripplot_kwargs = dict(
            size=10,
            jitter=0.2,
            linewidth=0.5,
            edgecolor='gray',
            alpha=0.5
        )
    default_stripplot_kwargs.update(stripplot_kwargs)

    # if hue is None, make the scatter color same as boxplot
    sns.stripplot(
        x=x, y=y,
        data=data,
        ax=ax,hue=hue,
        dodge=True,
        hue_order=ax.get_legend_handles_labels()[1],
        **default_stripplot_kwargs
    )
    
    for l in labeling:
        ax.scatter(x=0,y=0,s=0,label = l)
    return ax

def rank_strip_plot(
    data, x, s=None, c=None, 
    center=True, ax=None,
    platte = 'viridis',
    yticks_adjusts = 0.02,
    cbar_kwargs = {},
    legend_kwargs = {},
    **scatter_kwargs 
):
    """
    Plot a strip plot with rank order.
    """
    sns.set_theme(style="whitegrid")
    if ax is None:
        ax = plt.gca()
    
    
    data['y'] = data[x].rank()
    data = data.sort_values('y')

    max_s = 150
    min_s = 10
    if s is not None:
        # noralized point size
        data['s'] = (data[s] - data[s].min()) / (data[s].max() - data[s].min()) * max_s
        data['s'] += min_s

    ax.scatter(
        x=data[x], y=data['y'],
        s = data['s'] if s is not None else None, 
        c = data[c] if c is not None else None,
        cmap = platte,
        **scatter_kwargs
    )

    # make the y axis in the center
    center_loc = data[x].median()
    sns.despine(bottom=True)
    if center:
        ax.spines['left'].set_position(('data', center_loc))
        # ax.spines['left'].set_position(('axes', 0.5))

    ax.set_yticks(
        range(1,data.shape[0]+1)
    )
    
    ax.set_yticklabels(data.index)

    # for x > center, make the yticklabel in the right
    # for x < center, make the yticklabel in the left
    for i in ax.get_yticklabels():
        xi = data.loc[i.get_text(),x]
        if xi < center_loc:
            i.set_horizontalalignment('left')
            i.set_x(0.02+yticks_adjusts)
        else:
            i.set_horizontalalignment('right')
            i.set_x(-0.02+yticks_adjusts)

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # add legends for size and color.
    # 1. colorbar
    cbar = None
    if c is not None:
        cbar = plt.colorbar(
            cm.ScalarMappable(cmap=platte),
            ax=ax,
            anchor=(0.1,0.15),
            # change length of colorbar
            shrink=0.5, 
            ## change width of colorbar
            aspect = 10,
            **cbar_kwargs
            # ticks=[data[c].min(), data[c].max()]
        )
        cbar.set_label(c)
    # 2. size
    leg = None
    if s is not None:
        sseq = np.linspace(data['s'].min(),data['s'].max(),5)
        leg = ax.legend(
            handles=[
                ax.scatter([],[],s=i,c='.7',label=i) for i in sseq
            ],
            labels=[f'{i:.1f}' for i in np.linspace(data[s].min(),data[s].max(),5)],
            loc='upper right',
            bbox_to_anchor=(1.3, 1.0),
            title=s, frameon=False, **legend_kwargs
        )


    return {
        'ax':ax,
        'cbar':cbar,
        'leg':leg
    }

def paired_stacked_bar(data, width=0.5, cmap = 'tab20',ax=None, sep = 0.2, cmap_sequential=False):
    if ax is None:
        ax = plt.gca()

    # get x locations; 
    # if same group, xlocs +0.5 for each bar; else +1 for each bar
    xlocs = [0]
    for i in range(1, len(data.index)):
        if data.index[i].startswith(data.index[i-1].split('_')[0]):
            xlocs.append(xlocs[-1]+width)
        else:
            xlocs.append(xlocs[-1]+width+sep)

    # plot
    bottom = np.zeros(len(data.index))
    for i in range(ni:=len(data.columns)):
        color_i = (i+1) / (ni+1) if cmap_sequential else i
        ax.bar(
            xlocs, 
            data.iloc[:,i], 
            bottom=bottom, 
            label=data.columns[i],
            width=width,
            color=plt.cm.get_cmap(cmap)(color_i),
            # set edge width to 0 to make it smooth
            edgecolor='none'
        )
        bottom += data.iloc[:,i]

    ax.set_xticks(xlocs)
    ax.set_ylabel('Percentage')
    ax.set_xticklabels(data.index)
    ax.set_xlim(0-width, xlocs[-1]+width)

    return ax

def plot_categories_as_colorblocks(
    obs:pd.DataFrame, groupby:list,
    sorted=True,bar_width=0.2, sep=0.05,
    # plot args
    ax=None, cmap = 'tab20b',orientation='vertical',
    show = 'cat', # or 'number' or None, show the color blocks or not
    text_kwargs = dict(),
    **kwargs
):
    # like stack bar plot
    # but only one bar for each category
    # and the bar is colored by the category

    if ax is None:
        ax = plt.gca()
    palette = matplotlib.colormaps[cmap]
    # set x_loc, width*n + sep 
    x_loc = np.arange(len(groupby)) * (bar_width + sep)
    default_kwargs = dict(
        fontsize = 12,
        horizontalalignment='center',
        verticalalignment='center',
        rotation = 90 if orientation == 'vertical' else 0
        )
    default_kwargs.update(text_kwargs)

    # TODO.
    # dat = obs[groupby].value_counts()
    # if sorted:
    #     dat = dat.sort_index()
    # dat = dat.reset_index()


    for x,cat in zip(x_loc,groupby):
        dat = obs[cat].value_counts()
        if sorted:
            dat = dat.sort_index()
        dat_n = dat.copy()
        dat = dat / dat.sum()
        dat_cumsum = dat.cumsum()
        bottom = 0
        
        if orientation == 'vertical':
            for i in range(len(dat)):
                block_color = palette(i)
                r,g,b,a = block_color
                text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                ax.bar(
                    x, dat[i], 
                    width=bar_width,
                    bottom=bottom,
                    color=block_color,
                    edgecolor='none',
                    **kwargs
                )
                bottom += dat[i]
                if show == 'cat':
                    ax.text(
                        x, dat_cumsum[i] - dat[i]/2,
                        dat.index[i],
                        color=text_color,
                        **default_kwargs
                    )
                elif show == 'number':
                    ax.text(
                        x, dat_cumsum[i] - dat[i]/2,
                        f'{dat_n[i]:.2f}',
                        color=text_color,
                        **default_kwargs
                    )
        elif orientation == 'horizontal':
            for i in range(len(dat)):
                block_color = palette(i)
                r,g,b,a = block_color
                text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                ax.barh(
                    x, dat[i], 
                    left=bottom,
                    color=block_color,
                    edgecolor='none',
                    height=bar_width,
                    **kwargs
                )
                bottom += dat[i]
                if show == 'cat':
                    ax.text(
                        dat_cumsum[i] - dat[i]/2, x,
                        dat.index[i],
                        color=text_color,
                        **default_kwargs
                    )
                elif show == 'number':
                    ax.text(
                        dat_cumsum[i] - dat[i]/2, x,
                        f'{dat_n[i]:.2f}',
                        color=text_color,
                        **default_kwargs
                    )            
        else:
            raise ValueError(f'orientation should be vertical or horizontal, not {orientation}')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    return ax

if __name__=="__main__":
    import numpy as np
    import pandas as pd

    # test for the volcano plot
    # ---------------------------
    # generate random data
    x = np.random.normal(loc=0,scale=2,size=1000)
    y = np.random.random(size=1000) * 10
    df = pd.DataFrame({'x':x,'y':y})
    df.index = [f'x{i}' for i in range(1000)]

    plot_volcano(
        df=df,x='x',y='y',
        xthres=1,ythres=2,
        label_items=[
            'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'
            ])

    # test for the Enrichr_bar_group
    # ---------------------------
    # generate random data
    data = pd.DataFrame(
        np.random.random(size=(100,2)),
        columns=['Odds Ratio','Adjusted P-value']
    )
    data['Gene_set'] = np.random.choice(['GO','KEGG','Reactome'],size=100)
    data['Term'] = data['Gene_set'] + pd.Series(np.arange(100)).astype(str)
    data = data.sort_values(by='Adjusted P-value')
    data.index = data['Term']

    Enrichr_bar_group(data=data)

    # test for the rank_strip_plot
    # ---------------------------
    # generate random data
    data = pd.DataFrame(
        columns=['x','s','c'],
        data=np.random.random(size=(15,3)),
        dtype=np.float64
    )
    data.index = [f'xxxx{i}' for i in range(15)]
    rank_strip_plot(data, x='x', s='s',c='c')

    # test for the annote_line
    # ---------------------------
    # generate random data
    data = pd.DataFrame(
        columns=['x','y'],
        data=np.random.random(size=(15,2)),
        dtype=np.float64
    )
    data.index = [f'xxxx{i}' for i in range(15)]
    data['x'] = data['x'].rank()

    f, ax = plt.subplots()
    sns.barplot(data=data,x='x',y='y', ax=ax)
    annote_line(
        ax=ax,
        s='*',start=(0,0.5),end=(1,0.5),color='k',
        ygap= -0.015, fontsize=12,
        lw = 1
    )
    f

    # test for the plot_categories_as_colorblocks
    # ---------------------------
    # generate random data ; discrete data
    data = pd.DataFrame(
        columns=['x','y'],
        data=np.random.choice(['a','b','c'],size=(100,2), p=[0.1,0.2,0.7]),
        dtype=str
    )
    f, ax = plt.subplots(figsize=(8,1))
    plot_categories_as_colorblocks(
        data, groupby=['x','y'], ax=ax, orientation='horizontal',
    )