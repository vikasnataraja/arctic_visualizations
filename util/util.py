import os
import platform
import rasterio
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
from PIL import Image
import pyproj
import cartopy.crs as ccrs


# filenames of the satellite images
satellite_fnames = {
    '20240528': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-05-28-164000Z',},

    '20240530': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-05-30-144500Z',},

    '20240531': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-05-31-152500Z',},

    '20240603': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-06-03-141000Z',},

    '20240605': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-06-05-171000Z',},

    '20240606': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-06-06-161000Z',},

    '20240607': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-06-07-183000Z',},

    '20240610': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-06-10-171500Z',},

    '20240611': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-06-11-175500Z'},

    '20240613': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-06-13-142000Z',},

    '20240725': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-07-25-162000Z',},

    '20240729': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-07-29-154500Z',},

    '20240730': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-07-30-131000Z',},

    '20240801': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-08-01-143000Z',},

    '20240802': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-08-02-151000Z',},

    '20240807': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-08-07-165500Z',},

    '20240808': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-08-08-155500Z',},

    '20240809': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-08-09-181500Z',},

    '20240815': {'FalseColor367':'MODIS-TERRA_FalseColor367_2024-08-15-172000Z',},

}

parent_dir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

def format_time(total_seconds, format=None):
    """
    Convert seconds to hours, minutes, seconds, and milliseconds.

    Args:
    ----
        - total_seconds (int): The total number of seconds to convert.
        - format (string): if 'string' then will be formatted to return a string,
                           otherwise all the elements will be returned

    Returns:
    -------
        - A string or tuple containing hours, minutes, seconds, and milliseconds.
    """
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = (total_seconds - int(total_seconds)) * 1000

    if format == 'string':
        return '{}:{}:{}.{}'.format(int(hours), int(minutes), int(seconds), int(milliseconds))

    # otherwise return the whole thing
    return (int(hours), int(minutes), int(seconds), int(milliseconds))


def get_cpu_processes():

    # for local machines
    if (platform.uname().node == 'macbook') or (platform.uname().system == 'Darwin') or (platform.uname().system == 'Windows'):
        cores = int(multiprocessing.cpu_count()/4)

    else: # meant for supercomputers
        cores = int(multiprocessing.cpu_count())

    return cores


def load_sic(ymd):
    # read sea ice data file and lat-lons
    fsic = SD(os.path.join(parent_dir, 'data/sic_amsr2_bremen/{}/asi-AMSR2-n3125-{}-v5.4.hdf'.format(ymd, ymd)), SDC.READ)
    fgeo = SD(os.path.join(parent_dir, 'data/sic_amsr2_bremen/LongitudeLatitudeGrid-n3125-ArcticOcean.hdf'), SDC.READ)

    # AMSR2 Sea Ice Concentration
    sic = fsic.select('ASI Ice Concentration')[:]
    lon = fgeo.select('Longitudes')[:]
    lat = fgeo.select('Latitudes')[:]
    # lon = change_range(lon, -180, 180) # change from 0-360 to -180 to 180

    # mask nans and non-positive sic
    sic = np.ma.masked_where(np.isnan(sic) | (sic <= 0), sic)
    fsic.end()
    fgeo.end()

    return lon, lat, sic

def load_geotiff(filepath):

    dset = rasterio.open(filepath)

    band1 = dset.read(1)
    band2 = dset.read(2)
    band3 = dset.read(3)
    land_tiff = np.stack([band1, band2, band3], axis=-1)
    dset.close()

    return land_tiff


def load_land_feature(type='natural'):
    # load into memory ~ 700mb each
    if (type is not None) and (type.lower() == 'natural'):
        land_tiff_natural = load_geotiff(os.path.join(parent_dir, 'data/shapefiles/natural_earth_data/10m_natural_earth_relief/NE1_HR_LC_SR.tif'))
        return land_tiff_natural

    elif (type is not None) and ((type.lower() == 'topo') or (type.lower() == 'hypso')):
        land_tiff_hypsometric = load_geotiff(os.path.join(parent_dir, 'data/shapefiles/natural_earth_data/10m_natural_earth_relief_hypsometric/HYP_HR_SR.tif'))
        return land_tiff_hypsometric

    else:
        print('Message [load_land_feature]: `type` must be one of hypso, topo, or natural')
        return None


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

def load_blue_marble_imagery(modes, month):

    blue_marble_imgs = {}
    if isinstance(modes, list):

        # change to same case
        modes = [x.upper() for x in modes]

        for mode in modes:
            if mode in blue_marble_info.keys():
                if 'WORLD' == mode: # filename and image size is different for world
                    blue_marble_imgs[mode.upper()] = Image.open(os.path.join(parent_dir, 'data/blue_marble/2004_{}/world.topo.bathy.2004{}.3x21600x10800.png'.format(month, month)))

                else:
                    blue_marble_imgs[mode.upper()] = Image.open(os.path.join(parent_dir, 'data/blue_marble/2004_{}/world.topo.bathy.2004{}.3x21600x21600.{}.png'.format(month, month, mode.upper())))

    else:
        print('Message [get_blue_marble_imagery]: Must provide a list of `modes`. No action taken, returning an empty dictionary.')

    return blue_marble_imgs


def load_aircraft_graphic(mode, width):

    if (mode.upper() == 'P3') or (mode.upper() == 'P-3'):
        img = Image.open(os.path.join(parent_dir, 'data/assets/p3_red_transparent.png'))

    else:
        img = Image.open(os.path.join(parent_dir, 'data/assets/giii_blue_transparent.png'))

    img = img.resize((int(width * 1.2), width)) # retain 1.2 aspect ratio
    return img


def transform_extent(extent, source_ccrs, target_ccrs):
    transformer = pyproj.Transformer.from_proj(source_ccrs.proj4_init, target_ccrs.proj4_init, always_xy=True)
    x0, y0 = transformer.transform(extent[0], extent[2])
    x1, y1 = transformer.transform(extent[1], extent[3])
    return [x0, x1, y0, y1]


def load_satellite_image(ymd, mode='FalseColor367', satellite='Terra', instrument='MODIS'):
    # MODIS-TERRA_FalseColor367_2024-05-31-152500Z_(-877574.55,877574.55,-751452.90,963254.75)_(-80.0000,-30.0000,71.0000,88.0000)

    if mode == 'FalseColor367':
        data_dir = os.path.join(parent_dir, 'data/false_color_367')
    else:
        data_dir = os.path.join(parent_dir, 'data/true_color')

    all_files = os.listdir(data_dir)
    fbasename = list(filter(lambda x: x.startswith(satellite_fnames[ymd][mode]), all_files))[0]
    fname = os.path.join(data_dir, fbasename)

    xy_extent, geog_extent = get_extents(fname)

    img = plt.imread(fname)
    whitespace = (np.sum(img[:, :, :-1], axis=-1) == 3) # mask out whitespace
    whitespace_3d = np.stack([whitespace, whitespace, whitespace], axis=-1)

    img_3d = np.ma.masked_array(img[:, :, :-1], mask=whitespace_3d)
    np.ma.set_fill_value(img_3d, np.nan)

    ccrs_projection = ccrs.Orthographic(central_longitude=np.mean(geog_extent[:2]), central_latitude=np.mean(geog_extent[2:]))

    return img_3d, xy_extent, geog_extent, ccrs_projection


def get_extents(fname):
    xy_extent, geog_extent = os.path.basename(fname).split('_')[-2:]

    xy_extent = list(map(np.float64, xy_extent.strip('(').strip(')').split(',')))
    geog_extent = list(map(np.float16, os.path.splitext(geog_extent)[0].strip('(').strip(')').split(',')))

    return xy_extent, geog_extent
