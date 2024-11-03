import os
import platform
import rasterio
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
from PIL import Image

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
