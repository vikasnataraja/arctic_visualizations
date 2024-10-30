"""
plot_util.py

This module contains utility functions and colormaps for plotting data related to the SIF SAT project.
It includes functions for setting plot fonts and predefined colormaps.

Author: Vikas Nataraja
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as mplt
import matplotlib.font_manager as font_manager
import cartopy.crs as ccrs
import cmasher as cmr

# parent directory
parent_dir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

# set matplotlib style
# if 'MPL_STYLE' in os.environ.keys():
#     MPL_STYLE_PATH = os.environ['MPL_STYLE']

# elif 'MPL_STYLE_PATH' in os.environ.keys():
#     MPL_STYLE_PATH = os.environ['MPL_STYLE_PATH']

# else: # custom matplotlib stylesheet
    # MPL_STYLE_PATH = os.path.join(parent_dir, 'util/plotting_utils/sifsat_whitestyle.mplstyle')

MPL_STYLE_PATH = os.path.join(parent_dir, 'util/plotting_utils/sifsat_whitestyle.mplstyle')

def set_plot_fonts(plt, serif_style='sans-serif', font='Libre Franklin'):
    """
    Set the fonts for matplotlib plots.
    This function configures the font settings for matplotlib plots by adding
    custom fonts from a specified directory and setting the global font family.

    Args:
    ----
        plt : module
            The matplotlib.pyplot module.
        serif_style : str, optional
            The style of the font family to use, one of 'serif' or 'sans-serif'.
        font : str, optional
            The name of the font to use (default is 'Libre Franklin').
    """

    # look for fonts in the fonts directory
    all_fonts_dir = os.path.join(parent_dir, 'util/plotting_utils/fonts/')
    font_dir = [os.path.join(all_fonts_dir, f) for f in all_fonts_dir if os.path.isdir(os.path.join(all_fonts_dir, f))]
    # add font if available from user's environment
    # if 'MPL_FONT_DIR' in os.environ.keys():
    #     font_dir.append(os.environ['MPL_FONT_DIR'])

    # add all the fonts to matplotlib's library
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)

    # set font family globally (in-place for plt)
    plt.rc('font', **{'family':'{}'.format(serif_style),
                      '{}'.format(serif_style):['{}'.format(font)]})


############################## Cloud phase IR colormap ##############################
ctp_ir_cmap_arr = np.array([
                            [0.5, 0.5, 0.5, 1.], # clear
                            [0., 0., 0.55, 1.], # liquid
                            [0.75, 0.85, 0.95, 1.], # ice
                            [0.55, 0.55, 0.95, 1.], # mixed
                            [0., 0.95, 0.95, 1.]])# undet. phase
ctp_ir_cmap_ticklabels = np.array(["clear", "liquid", "ice", "mixed phase", "uncertain"])
ctp_ir_tick_locs = (np.arange(len(ctp_ir_cmap_ticklabels)) + 0.5)*(len(ctp_ir_cmap_ticklabels) - 1)/len(ctp_ir_cmap_ticklabels)
ctp_ir_cmap = matplotlib.colors.ListedColormap(ctp_ir_cmap_arr)
ctp_ir_cmap.set_bad("black", 1)


############################## Cloud phase SWIR/COP colormap ##############################
ctp_swir_cmap_arr = np.array([
                            #   [0, 0, 0, 1], # undet. mask
                              [0.5, 0.5, 0.5, 1.], # clear
                              [0., 0., 0.55, 1.], # liquid
                              [0.75, 0.85, 0.95, 1.], # ice
                              [0., 0.95, 0.95, 1.]])# no phase (liquid)
ctp_swir_cmap_ticklabels = np.array(["clear", "liquid", "ice", "uncertain"])
ctp_swir_tick_locs = (np.arange(len(ctp_swir_cmap_ticklabels)) + 0.5)*(len(ctp_swir_cmap_ticklabels) - 1)/len(ctp_swir_cmap_ticklabels)
ctp_swir_cmap = matplotlib.colors.ListedColormap(ctp_swir_cmap_arr)
# ctp_swir_cmap.set_bad("black", 1)


############################## Cloud top height colormap ##############################
cth_cmap_arr = np.array([[0., 0., 0., 1], # no retrieval
                        [0.5, 0.5, 0.5, 1], # clear
                        [0.05, 0.7, 0.95, 1], # low clouds
                        [0.65, 0.05, 0.3, 1.],  # mid clouds
                        [0.95, 0.95, 0.95, 1.]])    # high clouds
cth_cmap_ticklabels = ["undet.", "clear", "low\n0.1 - 2 km", "mid\n2 - 6 km", "high\n>=6 km"]
cth_tick_locs = (np.arange(len(cth_cmap_ticklabels)) + 0.5)*(len(cth_cmap_ticklabels) - 1)/len(cth_cmap_ticklabels)
cth_cmap = matplotlib.colors.ListedColormap(cth_cmap_arr)

############################## Cloud top temperature colormap ##############################
ctt_cmap_arr = np.array(list(mplt.get_cmap('Blues_r')(np.linspace(0, 0.8, 4))) + list(mplt.get_cmap('Reds')(np.linspace(0, 1, 4))))
ctt_cmap = matplotlib.colors.ListedColormap(ctt_cmap_arr)

############################## Sea ice concentration colormap ##############################
base_cmap = cmr.arctic
n_colors = 256
dark_colors = 230
dark = np.linspace(0.15, 0.9, dark_colors)
bright = np.linspace(0.9, 1.0, n_colors - dark_colors)
cmap_arr = np.hstack([dark, bright])
sic_cmap = matplotlib.colors.ListedColormap(base_cmap(cmap_arr))


arctic_cloud_cmap = 'RdBu_r'
arctic_cloud_alt_cmap = 'RdBu_r'

proj_data = ccrs.PlateCarree()
cfs_alert = (-62.3167, 82.5) # Station Alert
stn_nord  = (-16.6667, 81.6) # Station Nord
thule_pituffik = (-68.703056, 76.531111) # Pituffik Space Base
