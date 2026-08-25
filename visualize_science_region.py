
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from tqdm import tqdm

from util.plot_util import MPL_STYLE_PATH, set_plot_fonts
import util.util as viz_utils
from util.constants import inset_map_settings, flight_date_to_sf_dict, text_bg_colors

from visualize_flight_paths import add_esri_features, add_ancillary, add_aircraft_graphic, get_closest_datetime, get_time_indices, minimize_df, ccrs_nearside, ccrs_geog
from matplotlib.gridspec import GridSpec
import matplotlib.lines
import argparse
import sys
from visualize_flight_paths import get_filenames


def draw_extent(ax, extent, transform_crs=ccrs.PlateCarree()):

    x0, x1, y0, y1 = extent[0], extent[1], extent[2], extent[3]

    points = np.array([[x0, y0],
                       [x0, y1],
                       [x1, y1],
                       [x1, y0],
                       [x0, y0],
                      ])
    poly_patch = matplotlib.patches.Polygon(points, closed=True, edgecolor='green', fill=True, facecolor="purple", alpha=0.5, transform=transform_crs)

    ax.add_patch(poly_patch)


def add_globe_inset(ax_parent, inset_extent, bbox_to_anchor=(1.0, 1.0, 0.5, 0.5), width='25%', height='25%'):
    # create the inset axis
    ccrs_globe = ccrs.NearsidePerspective(central_longitude=-50, central_latitude=80, satellite_height=300e3)

    axins = inset_axes(ax_parent, loc='upper right',
                       width=width, height=height,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax_parent.transAxes,
                       axes_class=cartopy.mpl.geoaxes.GeoAxes,
                       axes_kwargs=dict(projection=ccrs_globe)
                      )
    # Add land, state borders, coastline, and country borders to inset map
    add_ancillary(axins, cartopy_black=False, coastline=True, land='default', ocean=True, gridlines=False, zorders={'ocean': 0, 'land': 0, 'coastline': 1})
    # draw the science-region extent (use PlateCarree transform for lon/lat coords)
    draw_extent(axins, inset_extent, transform_crs=ccrs.PlateCarree())
    axins.set_global()
def visualize_science_region(df_p3, df_g3=None, satellite=True, force_extent=False, view_extent=None, dx=None, dy=None, dt=None, outdir='data/viz_agu_zoomed/'):


    # for title
    start_dt_title_str = df_p3['datetime'].iloc[0].strftime('%Y-%m-%d')
    ymd_str = df_p3['datetime'].iloc[0].strftime('%Y%m%d')
    p3_date_str = df_p3['datetime'].iloc[0].strftime('%d %B, %Y')

    # save images in dirs with dates
    outdir_with_date = os.path.join(outdir, ymd_str)
    if not os.path.isdir(outdir_with_date):
        os.makedirs(outdir_with_date)

    if dx is None:
        dx = 10

    if dy is None:
        dy = 5

    start_dt = inset_map_settings[ymd_str]['start']
    end_dt = inset_map_settings[ymd_str]['end']

    if view_extent is None: # infer from limits of data
        if df_g3 is None:
            view_extent = [
                np.nanmin(df_p3['Longitude']) - 2,
                np.nanmax(df_p3['Longitude']) + 2,
                np.nanmin(df_p3['Latitude']) - 1,
                min(np.nanmax(df_p3['Latitude']) + 1, 89.5)
            ]
        else:
            view_extent = [
                min(np.nanmin(df_p3['Longitude']) - 2, np.nanmin(df_g3['Longitude']) - 2),
                max(np.nanmax(df_p3['Longitude']) + 2, np.nanmax(df_g3['Longitude']) + 2),
                min(np.nanmin(df_p3['Latitude']) - 1, np.nanmin(df_g3['Latitude']) - 1),
                min(np.nanmax(df_p3['Latitude']) + 1, np.nanmax(df_g3['Latitude']) + 1, 89.5)
            ]

    df_p3 = minimize_df(df_p3, 'P3')
    df_g3 = minimize_df(df_g3, 'G3')

    # default sampling interval
    if dt is None:
        dt = 60

    dt_idx = get_time_indices(df_p3, dt)
    img_p3 = viz_utils.load_aircraft_graphic(mode='P3', width=25)
    img_g3 = viz_utils.load_aircraft_graphic(mode='G3', width=20)

    for i_p3 in tqdm(dt_idx, total=dt_idx.size):

        p3_time = df_p3['datetime'].iloc[i_p3]

        if isinstance(p3_time, pd.Timestamp):
            p3_time = p3_time.to_pydatetime()

        elif isinstance(p3_time, np.datetime64):
            p3_time = viz_utils.np_to_python_datetime(p3_time)

        # if not (start_dt <= p3_time <= end_dt):
        #     continue

        p3_time_str = df_p3['datetime'].iloc[i_p3].strftime('%H:%MZ')
        fname_dt_str = p3_time.strftime('%Y%m%d_%H%MZ') # for image filename
        fname_out = os.path.join(outdir_with_date, fname_dt_str + '.png')
        if os.path.isfile(fname_out):
            continue

        # set internal extent to prevent size zoom out issues
        internal_extent = [view_extent[0] + 2, view_extent[1] - 2, view_extent[2] + 0.25, view_extent[3] - 0.25]
        plot_p3 = False
        if (internal_extent[0] < df_p3['Longitude'].iloc[i_p3] < internal_extent[1]) and (internal_extent[2] < df_p3['Latitude'].iloc[i_p3] < internal_extent[3]):
            plot_p3 = True

        plot_g3 = False
        if (df_g3 is not None) and (len(df_g3) > 0):
            _, i_g3 = get_closest_datetime(p3_time, df_g3)

            if (internal_extent[0] < df_g3['Longitude'].iloc[i_g3] < internal_extent[1]) and (internal_extent[2] < df_g3['Latitude'].iloc[i_g3] < internal_extent[3]):
                plot_g3 = True

        patches_legend = []
        if (not plot_p3) and (not plot_g3): # no need to plot
            continue

        elif (plot_p3) and (plot_g3):
            labels = ['P-3', 'G-III']
            colors = ['red', 'blue']

        elif (not plot_p3) and (plot_g3):
            labels = ['G-III']
            colors = ['blue']

        elif (plot_p3) and (not plot_g3):
            labels = ['P-3']
            colors = ['red']


        ######################################################
        fig = plt.figure(figsize=(12, 12))
        plt.style.use(MPL_STYLE_PATH)
        gs = GridSpec(1, 1, figure=fig)

        ax = fig.add_subplot(gs[0], projection=ccrs_geog)
        # add_ancillary(ax, cartopy_black=False, land='default', gridlines=True, dx=dx, dy=dy, title=None)
        # add_esri_features(ax, land=False, gridlines=True, coastline=True, ocean=False, dx=dx, dy=dy)

        if plot_p3:
            ax.plot(df_p3['Longitude'], df_p3['Latitude'], transform=ccrs.Geodetic(), linewidth=2, linestyle='--', color='gray', zorder=2)
            ax.plot(df_p3['Longitude'].iloc[:i_p3], df_p3['Latitude'].iloc[:i_p3], transform=ccrs.Geodetic(), linewidth=2, color='red', zorder=3)
            add_aircraft_graphic(ax, img_p3, df_p3['True_Heading'].iloc[i_p3], df_p3['Longitude'].iloc[i_p3], df_p3['Latitude'].iloc[i_p3], ccrs_geog, zorder=4)


        if plot_g3:
            _, i_g3 = get_closest_datetime(p3_time, df_g3)
            ax.plot(df_g3['Longitude'], df_g3['Latitude'], transform=ccrs.Geodetic(), linewidth=2, linestyle='--', color='gray', zorder=2)
            ax.plot(df_g3['Longitude'].iloc[:i_g3], df_g3['Latitude'].iloc[:i_g3], transform=ccrs.Geodetic(), linewidth=2, color='blue', zorder=3)
            add_aircraft_graphic(ax, img_g3, df_g3['True_Hdg'].iloc[i_g3], df_g3['Longitude'].iloc[i_g3], df_g3['Latitude'].iloc[i_g3], ccrs_geog, zorder=4)


        for i in range(len(labels)):
            # patches_legend.append(matplotlib.patches.Patch(color=colors[i], label=labels[i]))
            patches_legend.append(matplotlib.lines.Line2D([0], [0], color=colors[i], label=labels[i]))

        ax.legend(handles=patches_legend, loc='lower right', bbox_to_anchor=(1, 0.0), facecolor='white',
                    ncol=1, fancybox=True, shadow=False, frameon=True, prop={'size': 12})

        if satellite: # load satellite params
            sat_img, xy_extent_projection, geog_extent, ccrs_projection = viz_utils.load_satellite_image(ymd_str, mode='TrueColor')
            xy_extent_target = viz_utils.transform_extent(xy_extent_projection, ccrs_projection, ccrs_geog)

            ax.imshow(sat_img.filled(np.nan), extent=xy_extent_target, transform=ccrs_geog, zorder=1)

        if force_extent:
            ax.set_extent(view_extent, crs=ccrs_geog)

        globe_extent = ax.get_xbound() + ax.get_ybound()
        add_globe_inset(ax, globe_extent, bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), width='20%', height='20%')

        # add science flight number as a bbox
        ax.text(0.2, 0.06, 'NASA ARCSIX Science Flight {}'.format(flight_date_to_sf_dict[ymd_str][-2:]), fontweight=700, color='black', fontsize=16, ha="center", va="center", ma="center", transform=ax.transAxes, bbox=dict(facecolor=text_bg_colors[ymd_str], edgecolor='black', boxstyle='round, pad=0.5'), zorder=10)

        # add time
        ax.text(0.2, 0.02, '{} at {}'.format(p3_date_str, p3_time_str), fontweight="bold", color='white', fontsize=12, ha="center", va="center", ma="center", transform=ax.transAxes, bbox=dict(facecolor='black', edgecolor='black', boxstyle='round, pad=0.5'), zorder=10)

        ax.set_aspect('auto')

        # fig.set_facecolor('black')
        # save frame to file and close (no interactive display)
        fig.savefig(fname_out, dpi=300, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)

    return 1


def _read_iwg_file(iwg_file):
    df = pd.read_csv(iwg_file, index_col=0)
    if 'datetime' not in df.columns:
        if 'UTC_Time' in df.columns:
            df['datetime'] = df['UTC_Time']
        elif 'Date/Time' in df.columns:
            df['datetime'] = df['Date/Time']
        else:
            raise KeyError('datetime column not found in {}'.format(iwg_file))

    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate science-region flight-track frames')
    parser.add_argument('--iwg_dir', type=str, required=True, help='Path to directory containing IWG files')
    parser.add_argument('--date', type=str, required=True, help='Date in YYYYMMDD')
    parser.add_argument('--outdir', type=str, default='data/viz_agu_zoomed/', help='Output directory')
    parser.add_argument('--satellite', action='store_true', help='Overlay satellite true-color image')
    parser.add_argument('--force_extent', action='store_true', help='Force view extent to inferred region')
    parser.add_argument('--view_extent', type=str, default=None, help='Comma-separated lon0,lon1,lat0,lat1')
    parser.add_argument('--dt', type=int, default=60, help='Sampling interval in minutes')

    args = parser.parse_args()

    # find filenames
    p3_iwg_file, g3_iwg_file, lear_iwg_file = get_filenames(args)

    df_p3 = _read_iwg_file(p3_iwg_file)
    df_g3 = None
    if (g3_iwg_file is not None) and (os.path.isfile(g3_iwg_file)):
        df_g3 = _read_iwg_file(g3_iwg_file)

    view_extent = None
    if args.view_extent is not None:
        parts = args.view_extent.split(',')
        if len(parts) == 4:
            view_extent = [float(p) for p in parts]
        else:
            print('`--view_extent` must be four comma-separated values: lon0,lon1,lat0,lat1')
            sys.exit(1)

    # ensure outdir exists
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    # call the visualizer
    visualize_science_region(df_p3=df_p3, df_g3=df_g3, satellite=args.satellite, force_extent=args.force_extent, view_extent=view_extent, dt=args.dt, outdir=args.outdir)

