import os
import argparse
import pandas as pd
import datetime
import matplotlib
import platform
import rasterio
import multiprocessing
import cartopy
import numpy as np

import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt


import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from pyhdf.SD import SD, SDC

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from util.plot_util import MPL_STYLE_PATH, sic_cmap

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_cpu_processes():

    if (platform.uname().node == 'macbook') or (platform.uname().system == 'Darwin') or (platform.uname().system == 'Windows'):
        cores = int(multiprocessing.cpu_count()/4)

    else:
        cores = multiprocessing.cpu_count()

    return cores


def load_geotiff(filepath):

    dset = rasterio.open(filepath)

    band1 = dset.read(1)
    band2 = dset.read(2)
    band3 = dset.read(3)
    land_tiff = np.stack([band1, band2, band3], axis=-1)
    dset.close()

    return land_tiff



def add_ancillary(ax, title=None, scale=1, dx=20, dy=5, cartopy_black=False, ccrs_data=None, coastline=True, ocean=True, gridlines=True, land='topo', y_fontcolor='black'):
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
        colors = {'ocean':'black', 'land':'black', 'coastline':'black'}

    else:
        colors = {'ocean':'aliceblue', 'land':'#fcf4e8', 'coastline':'black'}

    if ocean:
        ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'), zorder=1, facecolor=colors['ocean'], edgecolor='none')

    if land is not None:

        if land == 'topo' or land == 'hypso':
            # load into memory ~ 700mb each
            land_tiff_hypsometric = load_geotiff('data/shapefiles/natural_earth_data/10m_natural_earth_relief_hypsometric/HYP_HR_SR.tif')
            ax.imshow(land_tiff_hypsometric, extent=[-180, 180, -90, 90], transform=ccrs_data, zorder=0)

        # TODO: Find a better way to pre-load this to avoid re-loading at every call
        elif land == 'natural':
            land_tiff_natural = load_geotiff('data/shapefiles/natural_earth_data/10m_natural_earth_relief/NE1_HR_LC_SR.tif')
            ax.imshow(land_tiff_natural, extent=[-180, 180, -90, 90], transform=ccrs_data, zorder=0)

        else:
            ax.add_feature(cartopy.feature.LAND.with_scale('50m'), zorder=0, facecolor=colors['land'], edgecolor='none')

    if coastline:
        ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'), zorder=2, edgecolor=colors['coastline'], linewidth=1, alpha=1)

    if gridlines:
        gl = ax.gridlines(linewidth=1.5, color='darkgray',
                    draw_labels=True, zorder=2, alpha=0.75, linestyle=(0, (1, 1)),
                    x_inline=False, y_inline=True, crs=ccrs_data)

        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, dx))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 90, dy))
        gl.xlabel_style = {'size': int(12 * scale), 'color': 'black'}
        gl.ylabel_style = {'size': int(12 * scale), 'color': y_fontcolor}
        gl.rotate_labels = False
        gl.top_labels    = False
        gl.xpadding = 7.5
        gl.ypadding = 7.5

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)


def df_doy_to_dt(doy_seconds):
    doy, seconds = doy_seconds.split('_')
    year_doy = '2024' + '_' + doy
    init_dt = datetime.datetime.strptime(year_doy, "%Y_%j")
    actual_dt = init_dt + datetime.timedelta(seconds=int(seconds))
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
    closest_dt = min(dt_list, key=lambda d: abs(d - dt))
    closest_dt_idx = df_secondary[df_secondary['datetime'] == closest_dt].index[0]
    return closest_dt, closest_dt_idx


def add_aircraft_graphic(ax, img, heading, lon, lat, source_ccrs, zorder):
    # transform the coordinates to the target projection
    x, y = ax.projection.transform_point(x=lon, y=lat, src_crs=source_ccrs)

    # rotate by heading but only if it is an actual number
    if not np.isnan(heading):
        img = img.rotate(heading, Image.BICUBIC)
    # create the AnnotationBbox
    ax.add_artist(AnnotationBbox(OffsetImage(img), (x, y), frameon=False, zorder=zorder))




def plot_flight_path(df_p3, df_g3=None, dx=20, dy=5, dt=5, overlay_sic=False, underlay_blue_marble=None, parallel=False):

    # p3 image graphic to be used as scatter marker
    img_p3 = Image.open('data/assets/p3_red_transparent.png')
    img_p3 = img_p3.resize((int(20*1.2), 20))

    if df_g3 is not None:
        # G-III image graphic to be used as scatter marker
        img_g3 = Image.open('data/assets/giii_blue_transparent.png')
        img_g3 = img_g3.resize((int(20*1.2), 20))


    dt_idx_p3 = get_time_indices(df_p3, dt) # P3 data sampled every dt
    # dt_idx_g3 = get_time_indices(df_g3, dt) # g3 data sampled every dt

    # use second index (not first in case there was an error) to use as reference date
    flight_path_dt = df_p3['datetime'][1].to_pydatetime()
    ymd = flight_path_dt.strftime('%Y%m%d')
    month = flight_path_dt.strftime('%m')

    # now for the extras
    if overlay_sic:
        # read sea ice data file and lat-lons
        fsic = SD('data/sic_amsr2_bremen/{}/asi-AMSR2-n3125-{}-v5.4.hdf'.format(ymd, ymd), SDC.READ)
        fgeo = SD('data/sic_amsr2_bremen/LongitudeLatitudeGrid-n3125-ArcticOcean.hdf', SDC.READ)

        # AMSR2 Sea Ice Concentration
        sic = fsic.select('ASI Ice Concentration')[:]
        lon = fgeo.select('Longitudes')[:]
        lat = fgeo.select('Latitudes')[:]
        # lon = change_range(lon, -180, 180) # change from 0-360 to -180 to 180

        # mask nans and non-positive sic
        sic = np.ma.masked_where(np.isnan(sic) | (sic <= 0), sic)
        fsic.end()
        fgeo.end()

    blue_marble_imgs = {}
    if underlay_blue_marble is not None:
        for type in underlay_blue_marble:
            if 'WORLD' == type.upper(): # filename and image size is different for world
                blue_marble_imgs[type.upper()] = plt.imread('data/blue_marble/2004_{}/world.topo.bathy.2004{}.3x21600x10800.png'.format(month, month))

            else:
                blue_marble_imgs[type.upper()] = plt.imread('data/blue_marble/2004_{}/world.topo.bathy.2004{}.3x21600x21600.{}.png'.format(month, month, type.upper()))

    if parallel:
        counts = np.arange(0, len(dt_idx_p3))
        p_args = create_args_parallel(df_p3, df_g3, dt_idx_p3, img_p3, img_g3, blue_marble_imgs, lon, lat, sic, dx, dy, counts)
        pool = multiprocessing.Pool(processes=get_cpu_processes())
        pool.starmap(make_figures, p_args)
        pool.close()

    else:
        for count, i_p3 in enumerate(dt_idx_p3):
            make_figures(df_p3, df_g3, i_p3, img_p3, img_g3, blue_marble_imgs, lon, lat, sic, dx, dy, count)
            break



def make_figures(df_p3, df_g3, i_p3, img_p3, img_g3, blue_marble_imgs, lon, lat, sic, dx, dy, count):
    """ Parallelized """

    p3_time = df_p3['datetime'][i_p3]

    if df_g3 is not None:
        _, i_g3 = get_closest_datetime(p3_time, df_g3)

    ####################################################################################
    fig = plt.figure(figsize=(20, 20))
    plt.style.use(MPL_STYLE_PATH)
    gs = GridSpec(1, 1, figure=fig)
    ax0 = fig.add_subplot(gs[0], projection=ccrs_nearside)
    add_ancillary(ax0, dx=dx, dy=dy, cartopy_black=True, coastline=True, land='natural', ocean=True, gridlines=False)

    # first P3
    # plot path in color until current pos; plot scatter with aircraft graphic at current pos; plot future path in transparent color
    ax0.plot(df_p3['Longitude'][:i_p3], df_p3['Latitude'][:i_p3], linewidth=2, transform=ccrs_geog, color='red', alpha=0.75, zorder=4)
    add_aircraft_graphic(ax0, img_p3, df_p3['True_Heading'][i_p3], df_p3['Longitude'][i_p3], df_p3['Latitude'][i_p3], ccrs_geog, zorder=4)
    ax0.plot(df_p3['Longitude'][i_p3:], df_p3['Latitude'][i_p3:], linewidth=2, transform=ccrs_geog, color='black', alpha=0.25, linestyle='--', zorder=4)

    # now G-III if needed
    if df_g3 is not None:
        # plot path in color until current pos; plot scatter with aircraft graphic at current pos; plot future path in transparent color
        ax0.plot(df_g3['Longitude'][:i_g3], df_g3['Latitude'][:i_g3], linewidth=2, transform=ccrs_geog, color='blue', alpha=0.75, zorder=4)
        add_aircraft_graphic(ax0, img_g3, df_g3['True_Hdg'][i_g3], df_g3['Longitude'][i_g3], df_g3['Latitude'][i_g3], ccrs_geog, zorder=4)
        ax0.plot(df_g3['Longitude'][i_g3:], df_g3['Latitude'][i_g3:], linewidth=2, transform=ccrs_geog, color='black', alpha=0.25, linestyle='--', zorder=4)

    # plot blue marble images
    for key in blue_marble_imgs.keys():
        ax0.imshow(blue_marble_imgs[key], extent=blue_marble_info[key], transform=ccrs_geog, zorder=3)

    # plot sea ice concentration
    if sic is not None:
        ax0.pcolormesh(lon, lat, sic, transform=ccrs_geog, cmap=sic_cmap, shading='nearest', zorder=3)

    ax0.set_global()
    fname_out = 'data/viz_agu/test_{}.png'.format(count)
    fig.savefig(fname_out, dpi=300, bbox_inches='tight', pad_inches=0.15)
    print('Saved figure: ', fname_out)
    # plt.show()
    plt.close()


def create_args_parallel(df_p3, df_g3, i_p3, img_p3, img_g3, blue_marble_imgs, lon, lat, sic, dx, dy, counts):
    arg_list = []
    for i in range(len(i_p3)):
        mini_list = []

        mini_list.append(df_p3)
        mini_list.append(df_g3)
        mini_list.append(i_p3[i])
        mini_list.append(img_p3)
        mini_list.append(img_g3)
        mini_list.append(blue_marble_imgs)
        mini_list.append(lon)
        mini_list.append(lat)
        mini_list.append(sic)
        mini_list.append(dx)
        mini_list.append(dy)
        mini_list.append(counts[i])

        arg_list.append(mini_list)

    return arg_list

# blue marble extent
blue_marble_info = {'WORLD': [-180, 180, -90, 90],

                    'A1': [-180, -90, 0, 90],
                    'B1': [-90, 0, 0, 90],
                    'C1': [0, 90, 0, 90],
                    'D1': [90, 180, 0, 90],

                    'A2': [-180, -90, -90, 0],
                    'B2': [-90, 0, -90, 0],
                    'C2': [0, 90, -90, 0],
                    'D2': [90, 180, -90, 0],
                   }


ccrs_ortho = ccrs.Orthographic(central_longitude=-50, central_latitude=80)
ccrs_nearside = ccrs.NearsidePerspective(central_longitude=-50, central_latitude=80, satellite_height=500e3)
ccrs_geog = ccrs.PlateCarree()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, type=str, help='Date for which data will be visualized')
    parser.add_argument('--parallel', action='store_true',
                        help='Pass --parallel to enable parallelization of processing spread over multiple CPUs.\n')
    parser.add_argument('--overlay_sic', action='store_true',
                        help='Pass --overlay_sic to overlay sea ice concentration from the day to the plot\n')
    parser.add_argument('--dt', default=60, type=int, help='Sampling time interval in minutes i.e., plot every dt minutes.')

    args = parser.parse_args()

    flight_dt = datetime.datetime.strptime(args.date, '%Y%m%d')
    flight_dt_str = flight_dt.strftime('%Y%m%d')

    df_p3 = read_p3_iwg(fname='/Users/vikas/workspace/arctic/sif_sat/data/arcsix-field/{}/ARCSIX-MetNav_P3B_{}_RA.ict'.format(flight_dt_str, flight_dt_str), mts=False)
    df_g3 = read_g3_iwg(fname='/Users/vikas/workspace/arctic/sif_sat/data/arcsix-field/{}/GIII_{}.txt'.format(flight_dt_str, flight_dt_str), mts=True)
    plot_flight_path(df_p3, df_g3=df_g3, dx=20, dy=5, dt=50, overlay_sic=args.overlay_sic, underlay_blue_marble=None, parallel=args.parallel)
