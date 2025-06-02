import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd

class DoubleTriangle:
    def __init__(self, color, direct=1):
        self.color = color
        self.direct = direct

class DoubleTriangleHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x, y = handlebox.xdescent, handlebox.ydescent
        w, h = handlebox.width, handlebox.height
        if orig_handle.direct:
            triangle = mpatches.Polygon([[x+w, y], [x, y+h], [x, y]],
                                            facecolor=orig_handle.color,
                                            #  edgecolor='black', 
                                            lw=0, 
                                            transform=handlebox.get_transform())
        else:
            triangle = mpatches.Polygon([[x+w , y], [x + w, y + h ], [x, y+h]],
                                            facecolor=orig_handle.color,
                                            #   edgecolor='black', 
                                            lw=0, 
                                            transform=handlebox.get_transform())
        handlebox.add_artist(triangle)
        return triangle

def geom_rectriangle(
    data_lower:pd.DataFrame,data_upper:pd.DataFrame,
    ax=None,square=False,cmap=None,legend=True,
    uplowlabels=None,color_map=None
    ):

    # Transparent edge line of ax
    sns.set_theme(style="white")
    sns.set_style(style={
        'axes.edgecolor': '1.0',
        }
        )

    if ax is None:
        ax = plt.gca()

    assert data_lower.shape == data_upper.shape
    rowtickslabel = data_lower.index
    coltickslabel = data_lower.columns
    assert set(rowtickslabel.values.tolist()) == set(data_upper.index.values.tolist())
    assert set(coltickslabel.values.tolist()) == set(data_upper.columns.values.tolist())
    data_upper = data_upper.loc[rowtickslabel,coltickslabel]

    if (data_lower.dtypes != 'object').all():
        assert (data_upper.dtypes != 'object').all(), 'color_map is None, but data2 is not numeric'
        color_array1 = np.array(data_lower).flatten()
        color_array2 = np.array(data_upper).flatten()
        color_map = None
        cmap = plt.cm.get_cmap('viridis') if cmap is None else cmap
    else:
        assert (data_lower.dtypes == 'object').all(), 'color_map is not None, but data1 is numeric'
        assert (data_upper.dtypes == 'object').all(), 'color_map is not None, but data2 is numeric'
        if color_map is None:
            unique_items = set(data_lower.values.flatten().tolist() + data_upper.values.flatten().tolist())
            item_map = {item:i for i,item in enumerate(unique_items)}
            color_array1 = np.array([item_map[item] for item in data_lower.values.flatten().tolist()])
            color_array2 = np.array([item_map[item] for item in data_upper.values.flatten().tolist()])
            cmap = plt.get_cmap('tab20') if cmap is None else cmap
            color_map = {item:cmap(i) for item,i in item_map.items()} 
        else:
            item_map = {item:i for i,item in enumerate(color_map.keys())}
            unique_items = list(color_map.keys())
            color_array1 = np.array([item_map[item] for item in data_lower.values.flatten().tolist()])
            color_array2 = np.array([item_map[item] for item in data_upper.values.flatten().tolist()])            
            cmap = plt.cm.colors.ListedColormap(list(color_map.values()))
    N,M = data_lower.shape
    M+=1;N+=1
    x = np.arange(M)
    y = np.arange(N)
    X,Y = np.meshgrid(x,y)

    M-=1;N-=1
    triangles1 = [
        (i + j*(M+1), i+1 + j*(M+1), i + (j+1)*(M+1)) 
        for j in range(N) for i in range(M)
        ]
    triangles2 = [
        (i+1 + j*(M+1), i+1 + (j+1)*(M+1), i + (j+1)*(M+1))
         for j in range(N) for i in range(M)
        ]

    triang1 = Triangulation(X.flatten(),Y.flatten(),triangles1)
    triang2 = Triangulation(X.flatten(),Y.flatten(),triangles2)
    ax.tripcolor(triang1,color_array1,cmap=cmap,edgecolors='white')
    ax.tripcolor(triang2,color_array2,cmap=cmap,edgecolors='white')
    if square:
        ax.set_aspect('equal')
    ax.set_xlim(0,x.max())
    ax.set_ylim(0,y.max())
    ax.set_xticks(x[:-1]+0.5)
    ax.set_yticks(y[:-1]+0.5)
    ax.set_yticklabels(rowtickslabel)
    ax.set_xticklabels(coltickslabel,rotation=90)

    width,hight = ax.get_window_extent().size
    if legend:
        if color_map is not None:
            handles = [plt.Rectangle((0,0),1,1, color=color_map[item]) for item in unique_items]
            labels = list(unique_items)

            legend1=ax.legend(
                handles,labels,
                frameon=True,edgecolor='#ffffff',facecolor='#ffffff',
                bbox_to_anchor=(1.05, 0., 0.3, 0.5),
                framealpha=0,
                bbox_transform=ax.transAxes,
                )

            labels2 = ['upper','lower'] if uplowlabels is None else uplowlabels
            handles2 = [DoubleTriangle('.65', 1), DoubleTriangle('.65', 0)]
            legend2 = ax.legend(
                handles2, labels2, 
                frameon=True,edgecolor='#ffffff',facecolor='#ffffff',
                bbox_to_anchor=(1.03, 0.25,0.3,0.5),
                bbox_transform=ax.transAxes,
                framealpha=0,
                handler_map={DoubleTriangle: DoubleTriangleHandler()}
                )
            ax.add_artist(legend1)
        else:
            cbar = ax.figure.colorbar(
                plt.cm.ScalarMappable(cmap=cmap),
                ax=ax,
                ticks=np.linspace(0,1,5),
                orientation='vertical',
                fraction=0.046,
                pad=0.04,
                )
            tickslabel=np.linspace(color_array1.min(),color_array1.max(),5).round(2),
            cbar.ax.set_yticklabels(*tickslabel)
            # ticks invisible
            cbar.ax.tick_params(length=0)
    return ax,color_map

def set_data(sample, mut_data, gene, mut_type):
    sample = sample[:-1]
    if mut_data.loc[gene,sample] is not np.nan:
        mut_data.loc[gene,sample] = 'MultiHit'#mut_data.loc[gene,sample] + '/' + mut_type
    else:
        mut_data.loc[gene,sample] = mut_type

if __name__ == '__main__':
    target_genes="""TP53
    ZFHX4
    CSMD3
    LRP1B
    PIK3CA
    SPTA1
    ERBB2
    ARID1A
    CCNE1
    MYC
    KRAS""".split()

    import json
    sample_map = json.load(open('../../data/processed/sample_id_dict.json'))
    all_samples = list(sample_map.values())
    target_samples = set()
    for sample in all_samples:
        sample_id = sample[:-1]
        if (sample_id+'T' in all_samples) and (sample_id+'M' in all_samples):
            target_samples.add(sample_id)
    target_samples.remove('Syn_01')
    target_samples.remove('Syn_05')

    target_samples = list(target_samples)
    target_samples.sort()
    target_samples_id = pd.Series([sample+x for sample in target_samples for x in ['T','M']])
    mut_dataT = pd.DataFrame(columns=target_samples,index=target_genes)
    mut_dataM = pd.DataFrame(columns=target_samples,index=target_genes)
    
    cnv_data = pd.read_csv('../../reports/cnv/cnvvcf_annote_sep.csv')
    cnv = cnv_data[
        (cnv_data["GENE"].isin(target_genes)) &
        (cnv_data["Sample"].isin(target_samples_id))]

    snv_data = pd.read_csv('../../data/interim/Somatic_mutation_fixed.maf',sep='\t')
    snv_data_ = pd.read_csv('../../data/interim/Syn13.maf',sep='\t')
    use_columns = ['Hugo_Symbol','Tumor_Sample_Barcode','Variant_Type']
    snv = pd.concat([snv_data[use_columns],snv_data_[use_columns]])
    snv = snv[
        (snv['Hugo_Symbol'].isin(target_genes)) &
        (snv['Tumor_Sample_Barcode'].isin(target_samples_id))
        ]
    snv.loc[:,'Variant_Type'] = snv['Variant_Type'].map({'SNP':'SNV','DEL':'INDEL','INS':'INDEL'})
    cnv.loc[:,'Variant_Type'] = cnv['ALT'].map({'DEL':'Deletion','DUP':'Amplification'})

    for i in range(snv.shape[0]):
        gene = snv.iloc[i]['Hugo_Symbol']
        sample = snv.iloc[i]['Tumor_Sample_Barcode']
        mut_type = snv.iloc[i]['Variant_Type']
        if sample[-1] == 'T':
            set_data(sample, mut_dataT, gene, mut_type)
        else:
            set_data(sample, mut_dataM, gene, mut_type)
    
    for i in range(cnv.shape[0]):
        gene = cnv.iloc[i]['GENE']
        sample = cnv.iloc[i]['Sample']
        mut_type = cnv.iloc[i]['Variant_Type']
        if sample[-1] == 'T':
            set_data(sample, mut_dataT, gene, mut_type)
        else:
            set_data(sample, mut_dataM, gene, mut_type)

    mut_dataT = mut_dataT.fillna('NA').loc[target_genes[::-1],:]
    mut_dataM = mut_dataM.fillna('NA').loc[target_genes[::-1],:]

    f = plt.figure(dpi=300)
    geom_rectriangle(
        mut_dataT,
        mut_dataM,
        uplowlabels=['Primary','Metastases'],
        color_map={
            'SNV':'#3c6e3b',
            'INDEL':'#b28966',
            'Deletion':'#4980ff',
            'Amplification':'#8c4f4f',
            'MultiHit':'#262626',
            'NA':'.85',
        }
    )
    f.savefig('../../reports/geom_rectriangle.pdf',bbox_inches='tight')