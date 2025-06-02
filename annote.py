def change_text(t,n=1):
    words = t.split(' ')
    tx = ''
    while len(words) >n:
        tx += ' '.join(words[0:n]) + '\n'
        words = words[n:]
    tx += ' '.join(words)
    return tx

def annote_line(
        ax,
        s,start,end,
        color='b',xgap= 0,ygap=0,lw=4,
        fontsize=8,rotation=0
    ):
    an1 = ax.annotate(
        '', start, end,
        xycoords='data',
        horizontalalignment='left', 
        verticalalignment='top',
        annotation_clip=False,
        arrowprops=dict(arrowstyle='-', color=color,lw=lw,
        )
    )

    an1_text = ax.annotate(
        s, ((start[0]+end[0])/2-xgap, (start[1]+end[1])/2-ygap),
        xycoords='data',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=fontsize,
        annotation_clip=False,
        rotation=rotation,
    )
    return an1,an1_text

def adding_stat_annote(
    data, x, y, hue, ax,
    annote_thres = 0.05, test_func = None, pairs = None, 
    annotator_kwargs = {},
    **annote_kwargs
):
    """
    Available parameters for annotator_kwargs:
        plot: defining the plot type, default is 'boxplot', other options are 'barplot' and 'violinplot' and etc.

    Available parameters for annote_kwargs:
        test: defining the test type, default is 'Mann-Whitney', other options are 't-test' and 'Wilcoxon' and etc.
        pvalue_format: defining the pvalue format, default is {'fontsize':10, 'text_format':'simple', 'show_test_name':False}
            test_format: defining the test format, default is 'simple'. Other options are 'star'.
        line_width: defining the line width, default is 1
        verbose: defining the verbose, default is False 
    """
    from statannotations.Annotator import Annotator
    from itertools import combinations
    from scipy.stats import mannwhitneyu, ranksums

    if test_func is None:
        test_func = mannwhitneyu

    default_annote_kwargs = dict(
        test='Mann-Whitney',
        pvalue_format = dict(
            fontsize = 10, text_format = 'simple', show_test_name=False
        ),
        line_width = 1,
        verbose = False
    )
    default_annote_kwargs.update(annote_kwargs)

    if pairs is None and hue is not None:
        pairs = [] 
        for tgl in combinations(labels:=data[hue].unique(), r = 2):
            l1,l2 = tgl
            test_result = data.query(f"`{hue}` in @tgl").groupby(x).apply(
                lambda x: test_func(
                    *x.groupby(hue).apply(lambda x:x[y].tolist())
                ).pvalue
            )
            sig_cell_types = test_result.index[test_result < annote_thres]
            pairs += [((ct, l1), (ct, l2)) for ct in sig_cell_types]
    elif pairs is None and hue is None:
        pairs = []
        for tgl in combinations(labels:=data[x].unique(), r = 2):
            l1,l2 = tgl
            test_result = test_func(
                *data.query(f"`{x}` in @tgl").groupby(x).apply(lambda x:x[y].tolist())
            ).pvalue
            if test_result < annote_thres:
                pairs += [(l1, l2)]
    if len(pairs) == 0:
        raise ValueError('No significant pairs found. try to change the `annote_thres` or `test_func`.')

    annotator = Annotator(
        ax, pairs,
        data = data,
        x = x, y =y, hue = hue,
        **annotator_kwargs,
    )
    annotator.configure(**default_annote_kwargs) 
    annotator.apply_and_annotate()

