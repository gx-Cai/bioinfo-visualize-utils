import matplotlib.colors as mcolors
import colorsys
import numpy as np

def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def create_customized_cmap(
    name, colors, position=None, bit=False, reverse=False
):
    """
    returns a matplotlib colormap generated from a list of colors
    
    Parameters
    ----------
    name : str
        name of the colormap
    colors : list
        contains colors in hex, RGB, or RGBA
    position : list
        contains floats that correspond to each color in clist.
        Must be increasing and have length equal to that of clist.
        Default None
    bit : bool
        divide each value in the 'position' array by the largest 
        value so that all values are between 0 and 1. Default False.
    reverse : bool
        if True, reverse the order of the colors. Default False.
    """
    
    if reverse:
        colors = colors[::-1]
    
    # regular division returns floats, 
    # but we want integers:
    if bit:
        bit_rgb = [(int(np.r_[rgb][0]/255.0*100)/100., 
                    int(np.r_[rgb][1]/255.0*100)/100., 
                    int(np.r_[rgb][2]/255.0*100)/100.) 
                   for rgb in colors]
    else:
        bit_rgb = [(np.r_[rgb][0]/255.0, 
                    np.r_[rgb][1]/255.0, 
                    np.r_[rgb][2]/255.0) 
                   for rgb in colors]
    
    if position:
        if not len(position) == len(colors):
            raise ValueError("position length must be the same as colors")
        else:
            if not bit:
                position = [float(i)/max(position) for i in position]
    else:
        position = np.linspace(0,1,len(colors))
    
    #print(np.vstack((position, bit_rgb)).T)
    dictlist = {}
    for i,key in enumerate(['red', 'green', 'blue']):
        dictlist.update(
            {
                key: [(position[j], bit_rgb[j][i], bit_rgb[j][i]) 
                      for j in range(len(position))]
            }
            
        )
    print(dictlist)
    cmap = mcolors.LinearSegmentedColormap(name, dictlist)
    
    return cmap

