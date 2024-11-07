import os
import sys
import argparse
import pandas as pd
import datetime
import matplotlib
import multiprocessing
import cartopy
import numpy as np
from tqdm import tqdm

import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from util.plot_util import MPL_STYLE_PATH, sic_cmap, set_plot_fonts
import util.util as viz_utils

from util.constants import inset_map_settings, flight_date_to_sf_dict, text_bg_colors

import warnings
import platform

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
set_plot_fonts(plt, 'sans-serif', 'Libre Franklin') # set font prop in place for plt
plt.style.use(MPL_STYLE_PATH)



if not ((platform.uname().node == 'macbook') or (platform.uname().system == 'Darwin') or (platform.uname().system == 'Windows')):
    matplotlib.use('Agg') # for supercomputer only


def add_ancillary(ax, title=None, scale=1, dx=20, dy=5, cartopy_black=False, ccrs_data=None, coastline=True, ocean=True, gridlines=True, land=None, y_fontcolor='black', zorders={'land': 0, 'ocean': 1, 'coastline': 2, 'gridlines': 2}):
    """
    Adds ancillary features to the axis.
    """

    if ccrs_data is None:
        ccrs_data = ccrs.PlateCarree()

    # set title manually because of boundary
    if title is not None:
        # ax.text(0.5, 0.95, title, ha="center", color='white', fontsize=20, fontweight="bold", transform=ax.transAxes,
        #         bbox=dict(facecolor='black', boxstyle='round', pad=1))
        title_fontsize = int(18 * scale)

        ax.set_title(title, pad=7.5, fontsize=title_fontsize, fontweight="bold")

    if cartopy_black:
        colors = {'ocean':'black', 'land':'black', 'coastline':'black', 'title':'white', 'background':'black'}

    else:
        colors = {'ocean':'aliceblue', 'land':'#fcf4e8', 'coastline':'black', 'title':'black', 'background':'white'}

    if ocean:
        ax.add_feature(cartopy.feature.OCEAN.with_scale('10m'), zorder=zorders['ocean'], facecolor=colors['ocean'], edgecolor='none')

    if land is not None:
        if ((isinstance(land, bool)) and (land)) or ((isinstance(land, str)) and (land.lower() == 'default')): #land=True or land = 'default'
            ax.add_feature(cartopy.feature.LAND.with_scale('10m'), zorder=zorders['land'], facecolor=colors['land'], edgecolor='none')

        elif (isinstance(land, str)) and (land.lower() in ['topo', 'natural', 'hypso']): # load and then show, made for parallel non-sharing
            land_tiff = viz_utils.load_land_feature(land)
            ax.imshow(land_tiff, extent=[-180, 180, -90, 90], transform=ccrs_data, zorder=zorders['land'])

        elif isinstance(land, np.ndarray): # show pre-loaded array, for serialized runs
            ax.imshow(land, extent=[-180, 180, -90, 90], transform=ccrs_data, zorder=zorders['land'])


    if coastline:
        ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), zorder=zorders['coastline'], edgecolor=colors['coastline'], linewidth=1, alpha=1)

    if gridlines:
        gl = ax.gridlines(linewidth=1.5, color='darkgray',
                    draw_labels=True, zorder=zorders['gridlines'], alpha=0.75, linestyle=(0, (1, 1)),
                    x_inline=False, y_inline=True, crs=ccrs_data)

        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, dx))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 90, dy))
        gl.xlabel_style = {'size': int(12 * scale), 'color': colors['title']}
        gl.ylabel_style = {'size': int(12 * scale), 'color': y_fontcolor}
        gl.rotate_labels = False
        gl.top_labels    = False
        gl.xpadding = 7.5
        gl.ypadding = 7.5

    for spine in ax.spines.values():
        if cartopy_black:
            spine.set_edgecolor('white')
        else:
            spine.set_edgecolor('black')

        spine.set_linewidth(1)


def df_doy_to_dt(doy_seconds):
    doy, seconds = doy_seconds.split('_')
    year_doy = '2024' + '_' + doy
    try:
        init_dt = datetime.datetime.strptime(year_doy, "%Y_%j")
        actual_dt = init_dt + datetime.timedelta(seconds=int(seconds))

    except Exception as err: #TODO: Change this to a class to fix this issue permanently
        print(err)
        init_dt = datetime.datetime(2024, 8, 1) # temporary fix
        actual_dt = init_dt + datetime.timedelta(seconds=1e4)

    return actual_dt

def df_timestamp_to_dt(timestamp):
    # EXAMPLE: 2024-05-31T18:37:26.004Z
    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    return dt

def read_p3_iwg(fname, mts=False):

    with open(fname, 'r') as fp:
        dat = fp.readlines()

    data_txt = [x.strip().split(',') for x in dat]

    if mts:

        if data_txt[0][0] != 'HEADER':
            columns = ['HEADER','TimeStamp','Latitude','Longitude','GPS MSL Altitude','WGS84 Altitude','Pressure Altitude','Radar Altitude','Ground Speed','True Airspeed','Indicated Airspeed','Mach Number','Vertical Velocity','True Heading','Track','Drift','Pitch','Roll','Side Slip','Angle of Attack','Ambient Temp','Dew Point','Total Air Temp','Static Press','Dynamic Press','Cabin Press','Wind Speed','Wind Direction','Vertical Wind Speed','Solar Zenith Angle','Sun Elevation Aircraft','Sun Azimuth Ground','Sun Azimuth Aircraft']
        else:
            columns = data_txt[0]

        # if data_txt[0]
        df = pd.DataFrame(data_txt[1:], columns=columns)
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.fillna(value=np.nan)

        float_cols = ['Latitude', 'Longitude', 'GPS MSL Altitude',
               'WGS84 Altitude', 'Pressure Altitude', 'Radar Altitude', 'Ground Speed',
               'True Airspeed', 'Indicated Airspeed', 'Mach Number',
               'Vertical Velocity', 'True Heading', 'Track', 'Drift', 'Pitch', 'Roll',
               'Side Slip', 'Angle of Attack', 'Ambient Temp', 'Dew Point',
               'Total Air Temp', 'Static Press', 'Dynamic Press', 'Cabin Press',
               'Wind Speed', 'Wind Direction', 'Vertical Wind Speed',
               'Solar Zenith Angle', 'Sun Elevation Aircraft', 'Sun Azimuth Ground',
               'Sun Azimuth Aircraft']

        float_dtype  = ['float64'] * len(float_cols)
        float_dtype_dict = dict(zip(float_cols, float_dtype))

        str_cols = ['HEADER', 'TimeStamp']
        str_dtype = ['str'] * len(str_cols)
        str_dtype_dict = dict(zip(str_cols, str_dtype))

        df = df.astype({**float_dtype_dict, **str_dtype_dict})
        df = df.reset_index(drop=True)
        df['datetime'] = df['TimeStamp'].apply(df_timestamp_to_dt)
        df = df.sort_values('datetime')
        return df

    else:
        try:
            # get the last one
            match_idx = [i for i, lstring in enumerate(data_txt) if 'Time_Start' in lstring][-1]
            df = pd.DataFrame(data_txt[match_idx + 1:], columns=data_txt[match_idx])

        except Exception as err:
            print(err)
            columns = ['Time_Start','Day_Of_Year','Latitude','Longitude','GPS_Altitude','Pressure_Altitude','Radar_Altitude','Ground_Speed','True_Air_Speed','Indicated_Air_Speed','Mach_Number','Vertical_Speed','True_Heading','Track_Angle','Drift_Angle','Pitch_Angle','Roll_Angle','Static_Air_Temp','Potential_Temp','Dew_Point','Total_Air_Temp','IR_Surf_Temp','Static_Pressure','Cabin_Pressure','Wind_Speed','Wind_Direction','U','V','Solar_Zenith_Angle','Aircraft_Sun_Elevation','Sun_Azimuth','Aircraft_Sun_Azimuth','Mixing_Ratio','Part_Press_Water_Vapor','Sat_Vapor_Press_H2O','Sat_Vapor_Press_Ice','Relative_Humidity']

            df = pd.DataFrame(data_txt[1:], columns=columns)

        df = df.fillna(value=np.nan)

        float_cols = ['Latitude', 'Longitude', 'GPS_Altitude',
               'Pressure_Altitude', 'Radar_Altitude', 'Ground_Speed', 'True_Air_Speed',
               'Indicated_Air_Speed', 'Mach_Number', 'Vertical_Speed', 'True_Heading',
               'Track_Angle', 'Drift_Angle', 'Pitch_Angle', 'Roll_Angle',
               'Static_Air_Temp', 'Potential_Temp', 'Dew/Frost_Point',
               'Total_Air_Temp', 'IR_Surf_Temp', 'Static_Pressure', 'Cabin_Pressure',
               'Wind_Speed', 'Wind_Direction', 'U', 'V', 'Solar_Zenith_Angle',
               'Aircraft_Sun_Elevation', 'Sun_Azimuth', 'Aircraft_Sun_Azimuth',
               'Mixing_Ratio', 'Part_Press_Water_Vapor', 'Sat_Vapor_Press_H2O',
               'Sat_Vapor_Press_Ice', 'Relative_Humidity']

        float_dtype  = ['float64'] * len(float_cols)
        float_dtype_dict = dict(zip(float_cols, float_dtype))

        str_cols = ['Time_Start', 'Day_Of_Year']
        str_dtype = ['str'] * len(str_cols)
        str_dtype_dict = dict(zip(str_cols, str_dtype))

        df = df.astype({**float_dtype_dict, **str_dtype_dict})

        df['doy_seconds'] = df['Day_Of_Year'] + '_' + df['Time_Start']
        df['datetime'] = df['doy_seconds'].apply(df_doy_to_dt)

        # turn erroneous data into NaN
        df[(df == -9999.0) | (df == -8888.0) | (df == -7777.0)] = np.nan
        return df


def read_g3_iwg(fname, mts=False):

    with open(fname, 'r') as fp:
        dat = fp.readlines()

    data_txt = [x.strip().split(',') for x in dat]

    if mts:

        if data_txt[0][0] != 'IWG1':
            columns = ['IWG1', 'Date/Time', 'Latitude', 'Longitude', 'GPS MSL Alt', 'WGS 84 Alt', 'Pressure Alt', 'Radar Alt',
                       'Ground Speed', 'True Airspeed', 'Indicated Airspeed', 'Mach Number', 'Vert Velocity', 'True Hdg',
                       'Track', 'Drift', 'Pitch', 'Roll', 'Side slip', 'Angle of Attack', 'Ambient Temp', 'Dew Point',
                       'Total Air Temp', 'Static Press', 'Dynamic Press', 'Cabin Pressure', 'Wind Speed', 'Wind Dir',
                       'Vert Wind Spd', 'Solar Zenith', 'Sun Elev AC', 'Sun Az Grd', 'Sun Az AC']
        else:
            columns = data_txt[0]

        # if data_txt[0]
        df = pd.DataFrame(data_txt[1:], columns=columns)
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.fillna(value=np.nan)

        original_columns = df.columns
        new_columns = [i.replace(' ', '_') for i in original_columns] # add underscores
        df.columns = new_columns

        float_cols =  ['Latitude', 'Longitude', 'GPS_MSL_Alt', 'WGS_84_Alt', 'Pressure_Alt', 'Radar_Alt', 'Ground_Speed', 'True_Airspeed',
                       'Indicated_Airspeed', 'Mach_Number', 'Vert_Velocity', 'True_Hdg', 'Track', 'Drift', 'Pitch', 'Roll', 'Side_slip',
                       'Angle_of_Attack', 'Ambient_Temp', 'Dew_Point', 'Total_Air_Temp', 'Static_Press', 'Dynamic_Press', 'Cabin_Pressure',
                       'Wind_Speed', 'Wind_Dir', 'Vert_Wind_Spd', 'Solar_Zenith', 'Sun_Elev_AC', 'Sun_Az_Grd', 'Sun_Az_AC']

        float_dtype  = ['float64'] * len(float_cols)
        float_dtype_dict = dict(zip(float_cols, float_dtype))

        str_cols = ['IWG1', 'Date/Time']
        str_dtype = ['str'] * len(str_cols)
        str_dtype_dict = dict(zip(str_cols, str_dtype))

        df = df.astype({**float_dtype_dict, **str_dtype_dict})
        df = df.reset_index(drop=True)
        df['datetime'] = df['Date/Time'].apply(df_timestamp_to_dt)
        df = df.sort_values('datetime')
        df = df.reset_index(drop=True)
        return df


def report_memory_usage(arr):
    byte_usage = arr.nbytes
    if byte_usage < 1024:
        units = 'B'

    elif (byte_usage >= 1024) and (byte_usage < 1024**2): # kb
        byte_usage = np.round(byte_usage / 1024, 2)
        units = 'kB'

    elif (byte_usage >= 1024**2) and (byte_usage < 1024**3): # mb
        byte_usage = np.round(byte_usage / (1024**2), 2)
        units = 'mB'

    else: # gb
        byte_usage = np.round(byte_usage / (1024**3), 2)
        units = 'gB'

    print('{} {}'.format(byte_usage, units))


def change_range(arr, min_value, max_value):
    return (arr - min_value) % (max_value - min_value) + min_value


def get_time_indices(df, dt):
    # convert ns to s
    seconds = list(np.array((np.diff(df['datetime'])/1e9), dtype='int'))
    seconds.insert(0, 0) # to help with indexing and make array same size as df

    # convert to minutes
    mins = np.cumsum(seconds)/60.

    # indices we want
    dt_idx = np.where(mins % dt == 0.)[0]

    # add in the last index too to ensure we plot the last time index
    dt_idx = list(dt_idx)
    dt_idx.append(len(df) - 1)
    dt_idx = np.array(dt_idx)

    return dt_idx


def get_closest_datetime(dt, df_secondary):

    dt_list = list(df_secondary['datetime'])
    sample_time = dt_list[0]
    if isinstance(sample_time, pd.Timestamp):
        dt_list = [i.to_pydatetime() for i in dt_list]

    elif isinstance(sample_time, np.datetime64):
        dt_list = [np_to_python_datetime(i) for i in dt_list]

    closest_dt = min(dt_list, key=lambda d: abs(d - dt))
    closest_dt_idx = dt_list.index(closest_dt)
    return closest_dt, closest_dt_idx


def report_p3_dates(df_p3):
    # use second index (not first in case there was a misread header) to use as reference date
    p3_start_dt = df_p3['datetime'].iloc[1].to_pydatetime()
    p3_end_dt   = df_p3['datetime'].iloc[len(df_p3) - 1].to_pydatetime()
    p3_flight_duration = viz_utils.format_time((p3_end_dt - p3_start_dt).total_seconds(), format='string')
    ymd = p3_start_dt.strftime('%Y%m%d')
    month = p3_start_dt.strftime('%m')
    # report times
    print('Message [report_p3_dates]: P-3 flight: {} to {}, total duration = {}'.format(p3_start_dt.strftime('%Y-%m-%d_%H%MZ'), p3_end_dt.strftime('%Y-%m-%d_%H%MZ'), p3_flight_duration))

    return ymd, month


def report_g3_dates(df_g3):
    # report times
    g3_start_dt = df_g3['datetime'].iloc[1].to_pydatetime()
    g3_end_dt   = df_g3['datetime'].iloc[len(df_g3) - 1].to_pydatetime()
    g3_flight_duration = viz_utils.format_time((g3_end_dt - g3_start_dt).total_seconds(), format='string')
    print('Message [plot_flight_path]: G-III flight: {} to {}, total duration = {}'.format(g3_start_dt.strftime('%Y-%m-%d_%H%MZ'), g3_end_dt.strftime('%Y-%m-%d_%H%MZ'), g3_flight_duration))


def minimize_df(df, mode):
    """only keep the columns we need since it is a large dataset"""

    if df is None:
        return None

    if (mode.upper() == 'P3') or (mode.upper() == 'P-3'):
        keep_cols = ['Latitude', 'Longitude', 'True_Heading', 'Track_Angle', 'datetime']
        if len(df) > 1e4:
            df = df[::10] # reduces by a factor of 10
    else:
        keep_cols = ['Latitude', 'Longitude', 'True_Hdg',     'Track',       'datetime']

    df = df[keep_cols]
    return df


def np_to_python_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object
    Input:
      date - a np.datetime64 object
    Output:
      date - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
    py_date   = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    py_date   = py_date.replace(tzinfo=None) # so that timedelta does not raise an error
    return py_date

def create_dictionary(df, img, mode):

    data = {}
    if df is None:
        return data

    if (mode.upper() == 'P3') or (mode.upper() == 'P-3'):
        keep_cols = ['Latitude', 'Longitude', 'True_Heading', 'Track_Angle', 'datetime']
    else:
        keep_cols = ['Latitude', 'Longitude', 'True_Hdg',     'Track',       'datetime']

    for column in keep_cols:
        data[column] = df[column].values

    # also add image (Pillow Image object)
    data['img'] = img
    # data['img_shape'] = np.array(img).shape

    return data


def add_aircraft_graphic(ax, img, heading, lon, lat, source_ccrs, zorder):
    # transform the coordinates to the target projection
    x, y = ax.projection.transform_point(x=lon, y=lat, src_crs=source_ccrs)

    # rotate by heading but only if it is an actual number
    if not np.isnan(heading):
        img = img.rotate(heading, Image.BICUBIC)
    # create the AnnotationBbox
    ax.add_artist(AnnotationBbox(OffsetImage(img), (x, y), frameon=False, zorder=zorder))


def add_inset(ax_parent, inset_extent, p3_data, g3_data, i_p3, bbox_to_anchor, width='75%', height='60%'):
    """ Add inset to existing parent axis map"""

    p3_time = p3_data['datetime'][i_p3]

    if isinstance(p3_time, pd.Timestamp):
        p3_time = p3_time.to_pydatetime()

    elif isinstance(p3_time, np.datetime64):
        p3_time = np_to_python_datetime(p3_time)

    # only add the inset if either the P-3 or the G-III are within the region
    # whichever one is out of bounds will not be plotted within the inset

    plot_p3 = False
    if (inset_extent[0] < p3_data['Longitude'][i_p3] < inset_extent[1]) and (inset_extent[2] < p3_data['Latitude'][i_p3] < inset_extent[3]):
        plot_p3 = True

    plot_g3 = False
    if len(g3_data) > 0:
        _, i_g3 = get_closest_datetime(p3_time, g3_data)

        if (inset_extent[0] < g3_data['Longitude'][i_g3] < inset_extent[1]) and (inset_extent[2] < p3_data['Latitude'][i_g3] < inset_extent[3]):
            plot_g3 = True

    if (not plot_p3) and (not plot_g3): # no need to plot
        return 0


    # load satellite params
    sat_img, xy_extent_projection, geog_extent, ccrs_projection = viz_utils.load_satellite_image(p3_time.strftime('%Y%m%d'))
    xy_extent_target = viz_utils.transform_extent(xy_extent_projection, ccrs_projection, ccrs_nearside)

    # create the inset axis
    axins = inset_axes(ax_parent, width=width, height=height,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax_parent.transAxes,
                       axes_class=cartopy.mpl.geoaxes.GeoAxes,
                       axes_kwargs=dict(projection=ccrs_nearside)
                      )

    # Add land, state borders, coastline, and country borders to inset map
    add_ancillary(axins, cartopy_black=True, coastline=True, land=None, ocean=True, gridlines=False, zorders={'ocean': 0, 'coastline': 2})

    # add satellite image
    # axins.imshow(sat_img.filled(np.nan), extent=xy_extent_projection, transform=ccrs_projection, zorder=1)
    axins.imshow(sat_img.filled(np.nan), extent=xy_extent_target, transform=ccrs_nearside, zorder=1)

    # now plot inside
    if plot_p3:
        img_p3 = p3_data['img']
        # plot path in color until current pos; plot scatter with aircraft graphic at current pos; plot future path in transparent color
        axins.plot(p3_data['Longitude'], p3_data['Latitude'], linewidth=2, transform=ccrs_geog, color='black', alpha=0.25, linestyle='--', zorder=4)
        axins.plot(p3_data['Longitude'][:i_p3], p3_data['Latitude'][:i_p3], linewidth=2, transform=ccrs_geog, color='cyan', alpha=0.75, zorder=5)
        add_aircraft_graphic(axins, img_p3, p3_data['True_Heading'][i_p3], p3_data['Longitude'][i_p3], p3_data['Latitude'][i_p3], ccrs_geog, zorder=5)

    # now G-III if needed
    if plot_g3:
        _, i_g3 = get_closest_datetime(p3_time, g3_data)

        img_g3 = g3_data['img']
        # plot path in color until current pos; plot scatter with aircraft graphic at current pos; plot future path in transparent color
        axins.plot(g3_data['Longitude'], g3_data['Latitude'], linewidth=2, transform=ccrs_geog, color='black', alpha=0.25, linestyle='--', zorder=4)
        axins.plot(g3_data['Longitude'][:i_g3], g3_data['Latitude'][:i_g3], linewidth=2, transform=ccrs_geog, color='blue', alpha=0.75, zorder=5)
        add_aircraft_graphic(axins, img_g3, g3_data['True_Hdg'][i_g3], g3_data['Longitude'][i_g3], g3_data['Latitude'][i_g3], ccrs_geog, zorder=5)

    # Set the lat/lon limits of the inset map [x0, x1, y0, y1]
    axins.set_extent(inset_extent, ccrs_geog)

    # change border colors and width
    for spine in axins.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # add connectors from main map to inset
    _, connectors = ax_parent.indicate_inset_zoom(axins, edgecolor="black", linewidth=2, alpha=1, transform=ax_parent.transData)

    # highlight only a couple of connectors
    # 0: bottom left corner, 1: top left corner, 2: bottom right corner, 3: top right corner
    for i in np.arange(4):
        if i in inset_map_settings[p3_time.strftime('%Y%m%d')]['connectors']:
            connectors[i].set_visible(True)
            connectors[i].set_linewidth(2)
        else:
            connectors[i].set_visible(False)
    return 1


def plot_flight_path(df_p3, df_g3, outdir, overlay_sic, parallel, dt):

    df_p3 = minimize_df(df_p3, 'P3')
    df_g3 = minimize_df(df_g3, 'G3')

    if df_g3 is not None:
        print('Message [plot_flight_path]: # of samples\nP-3   = {}\nG-III = {}'.format(len(df_p3), len(df_g3)))
    else:
        print('Message [plot_flight_path]: # of samples\nP-3   = {}\nG-III = 0'.format(len(df_p3)))

    dt_idx_p3 = get_time_indices(df_p3, dt) # P3 data sampled every dt
    print('Message [plot_flight_path]: {} time steps will be visualized'.format(dt_idx_p3.size))

    # get dates and print a statement
    ymd, month = report_p3_dates(df_p3)

    # save images in dirs with dates
    outdir_with_date = os.path.join(outdir, ymd)
    if not os.path.isdir(outdir_with_date):
        os.makedirs(outdir_with_date)

    img_p3 = viz_utils.load_aircraft_graphic(mode='P3', width=25) # P-3 image graphic to be used as scatter marker

    if df_g3 is not None:
        img_g3 = viz_utils.load_aircraft_graphic(mode='G3', width=20) # G-III image graphic to be used as scatter marker
        report_g3_dates(df_g3)

    else:
        img_g3 = None # to prevent errors

    if overlay_sic:
        # read sea ice data file and lat-lons in delayed fashion
        sic_data = {}
        lon, lat, sic = viz_utils.load_sic(ymd)
        sic_data['lon'] = lon
        sic_data['lat'] = lat
        sic_data['sic'] = sic

    ############### start execution ###############
    p3_data = create_dictionary(df_p3, img_p3, 'P3')
    g3_data = create_dictionary(df_g3, img_g3, 'G3')

    if parallel:
        n_cores = viz_utils.get_cpu_processes()
        print('Message [plot_flight_path]: Processing will be spread across {} cores'.format(n_cores))

        with multiprocessing.Pool(processes=n_cores) as pool:
            pool.starmap(make_figures, [[outdir_with_date, p3_data, g3_data, i_p3, sic_data] for i_p3 in dt_idx_p3])

    else: # serially
        for count, i_p3 in tqdm(enumerate(dt_idx_p3), total=dt_idx_p3.size):
            _ = make_figures(outdir_with_date, p3_data, g3_data, i_p3, sic_data)


def make_figures(outdir, p3_data, g3_data, i_p3, sic_data):
    """ Parallelized """

    p3_time = p3_data['datetime'][i_p3]

    if isinstance(p3_time, pd.Timestamp):
        p3_time = p3_time.to_pydatetime()

    elif isinstance(p3_time, np.datetime64):
        p3_time = np_to_python_datetime(p3_time)

    p3_time_str = p3_time.strftime('%H:%MZ')
    p3_date_str = p3_time.strftime('%d %B, %Y')
    fname_dt_str = p3_time.strftime('%Y%m%d_%H%MZ') # for image filename
    ymd_str = p3_time.strftime('%Y%m%d') # for dictionary label for text

    fname_out = os.path.join(outdir, fname_dt_str + '.png')
    if os.path.isfile(fname_out):
        print('Message [make_figures]: Skipping {} as it already exists.'.format(fname_out))
        return 0

    title_str = 'NASA ARCSIX - Flight Path'
    credit_text = 'SIC Data from AMSR2/GCOM-W1 Spreen et al. (2008)\n\n'\
                  'Visualization by Vikas Nataraja/SSFR Team'

    ####################################################################################
    # print('Starting to create figure for {}'.format(p3_time_str))
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(1, 1, figure=fig)
    ax0 = fig.add_subplot(gs[0], projection=ccrs_nearside)
    add_ancillary(ax0, dx=20, dy=5, cartopy_black=True, coastline=True, land=land, ocean=True, gridlines=False)

    # first P3
    img_p3 = p3_data['img']
    # plot path in color until current pos; plot scatter with aircraft graphic at current pos; plot future path in transparent color
    ax0.plot(p3_data['Longitude'], p3_data['Latitude'], linewidth=2, transform=ccrs_geog, color='black', alpha=0.25, linestyle='--', zorder=4)
    ax0.plot(p3_data['Longitude'][:i_p3], p3_data['Latitude'][:i_p3], linewidth=2, transform=ccrs_geog, color='red', alpha=0.75, zorder=5)
    add_aircraft_graphic(ax0, img_p3, p3_data['True_Heading'][i_p3], p3_data['Longitude'][i_p3], p3_data['Latitude'][i_p3], ccrs_geog, zorder=5)

    # now G-III if it exists
    if len(g3_data) > 0:
        _, i_g3 = get_closest_datetime(p3_time, g3_data)
        img_g3 = g3_data['img']
        # plot path in color until current pos; plot scatter with aircraft graphic at current pos; plot future path in transparent color
        ax0.plot(g3_data['Longitude'], g3_data['Latitude'], linewidth=2, transform=ccrs_geog, color='black', alpha=0.25, linestyle='--', zorder=4)
        ax0.plot(g3_data['Longitude'][:i_g3], g3_data['Latitude'][:i_g3], linewidth=2, transform=ccrs_geog, color='blue', alpha=0.75, zorder=5)
        add_aircraft_graphic(ax0, img_g3, g3_data['True_Hdg'][i_g3], g3_data['Longitude'][i_g3], g3_data['Latitude'][i_g3], ccrs_geog, zorder=5)

    # plot blue marble images
    if len(blue_marble_imgs) > 0:
        for key in blue_marble_imgs.keys():
            ax0.imshow(blue_marble_imgs[key], extent=viz_utils.blue_marble_info[key], transform=ccrs_geog, zorder=3)

    # plot sea ice concentration
    if len(sic_data) > 0:
        ax0.pcolormesh(sic_data['lon'], sic_data['lat'], sic_data['sic'], transform=ccrs_geog, cmap=sic_cmap, shading='nearest', zorder=3, alpha=1)

    ax0.set_global()
    fig.set_facecolor('black') # for hyperwall

    # Add inset map if within focus region
    if inset_map_settings[ymd_str]['start'] <= p3_time <= inset_map_settings[ymd_str]['end']:

        add_inset(ax_parent=ax0, inset_extent=inset_map_settings[ymd_str]['extent'], p3_data=p3_data, g3_data=g3_data, i_p3=i_p3, bbox_to_anchor=(0.3, -0.05, 0.6, 0.6), width='75%', height='60%')

    # add science flight number as a bbox
    ax0.text(0.88, 0.05, 'NASA ARCSIX Science Flight {}'.format(flight_date_to_sf_dict[ymd_str][-2:]), fontweight="bold", color='black', fontsize=14, ha="center", va="center", ma="center", transform=ax0.transAxes, bbox=dict(facecolor=text_bg_colors[ymd_str], edgecolor='white', boxstyle='round, pad=0.5'))

    # add time
    ax0.text(0.88, 0.025, '{} at {}'.format(p3_date_str, p3_time_str), fontweight="bold", color='white', fontsize=14, ha="center", va="center", ma="center",
             transform=ax0.transAxes)

    # plt.show()
    fig.savefig(fname_out, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)

    # print('Saved figure: ', fname_out)
    return 1

def get_filenames(args):

    flight_dt = datetime.datetime.strptime(args.date, '%Y%m%d')
    flight_dt_str = flight_dt.strftime('%Y%m%d')

    if args.iwg_dir is not None:
        flight_dir = os.path.join(args.iwg_dir, flight_dt_str)
        p3_iwg_file, g3_iwg_file = None, None
        for flight_file in os.listdir(flight_dir):
            if ('ARCSIX-MetNav_P3B_{}'.format(flight_dt_str) in flight_file) and (flight_file.endswith('.ict')):
                p3_iwg_file = os.path.join(flight_dir, flight_file)

            if ('GIII_{}'.format(flight_dt_str) in flight_file) and (flight_file.endswith('.txt')):
                g3_iwg_file = os.path.join(flight_dir, flight_file)

        if p3_iwg_file is None: # do not proceed
            print('Could not find the P-3 MetNav IWG File. Exiting...')
            sys.exit()

        return p3_iwg_file, g3_iwg_file

    else:
        print('Must provide `--iwg_dir` to locate the P-3 MetNav IWG File inside the directory.')
        sys.exit()


ccrs_ortho = ccrs.Orthographic(central_longitude=-50, central_latitude=80)
ccrs_nearside = ccrs.NearsidePerspective(central_longitude=-50, central_latitude=80, satellite_height=500e3)
ccrs_geog = ccrs.PlateCarree()

if __name__ == '__main__':

    exec_start_dt = datetime.datetime.now() # to time the whole thing
    multiprocessing.set_start_method('fork')
    parser = argparse.ArgumentParser()
    parser.add_argument('--iwg_dir', type=str, help='Path to directory containing P-3 and G-III IWG/MetNav files')
    parser.add_argument('--outdir',  type=str, help='Path to directory where the images will be written to')
    parser.add_argument('--date',    type=str, help='Date for which data will be visualized')
    parser.add_argument('--parallel', action='store_true',
                        help='Pass --parallel to enable parallelization of processing spread over multiple CPUs.\n')
    parser.add_argument('--overlay_sic', action='store_true',
                        help='Pass --overlay_sic to overlay sea ice concentration from the day to the plot\n')
    parser.add_argument('--underlay_blue_marble', default=None, type=str,
                        help='Underlay blue marble imagery, one of `world`, `topo`\n')
    parser.add_argument('--dt', default=60, type=int, help='Sampling time interval in minutes i.e., plot every dt minutes.')

    args = parser.parse_args()


    p3_iwg_file, g3_iwg_file = get_filenames(args)
    df_p3 = read_p3_iwg(fname=p3_iwg_file, mts=False)

    # if G-III file exists, read it, else None

    if args.date == '20240528': # transit for G-III, skip
        df_g3 = None
        print('No G-III track will be plotted for {} since it was a transit day'.format(args.date))

    else:
        if (g3_iwg_file is not None) and (os.path.isfile(g3_iwg_file)):
            df_g3 = read_g3_iwg(fname=g3_iwg_file, mts=True)

        else:
            df_g3 = None
            print('No G-III track found for {}'.format(args.date))

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # get dates and add a print statement
    ymd, month = report_p3_dates(df_p3)

    ############### load blue marble imagery into a dictionary ###############
    if args.underlay_blue_marble is not None:
        blue_marble_imgs = viz_utils.load_blue_marble_imagery(args.underlay_blue_marble, month)
        land = None

    else: # use other land features instead
        blue_marble_imgs = {}
        land = viz_utils.load_land_feature(type='natural')

    plot_flight_path(df_p3=df_p3, df_g3=df_g3, dt=args.dt, outdir=args.outdir, overlay_sic=args.overlay_sic, parallel=args.parallel)
    exec_stop_dt = datetime.datetime.now() # to time sdown
    exec_total_time = exec_stop_dt - exec_start_dt
    sdown_hrs, sdown_mins, sdown_secs, sdown_millisecs = viz_utils.format_time(exec_total_time.total_seconds())
    print('\n\nTotal Execution Time: {}:{}:{}.{}\n\n'.format(sdown_hrs, sdown_mins, sdown_secs, sdown_millisecs))
