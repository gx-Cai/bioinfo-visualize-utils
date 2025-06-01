# coding: utf-8

# ipython notebook requires this
# %matplotlib inline

# python console requires this
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import venn

labels, set_collections = venn.get_labels([range(10), range(5, 15)], fill=['number', 'logic']) #fill: ["number"|"logic"|"percent"|"item"]
fig, ax = venn.venn2(labels, names=['list 1', 'list 2'])
fig.savefig('venn2.png', bbox_inches='tight')
plt.close()

