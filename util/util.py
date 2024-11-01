import os
import platform
import rasterio
import multiprocessing
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

def format_time(total_seconds):
    """
    Convert seconds to hours, minutes, seconds, and milliseconds.

    Parameters:
    - total_seconds: The total number of seconds to convert.

    Returns:
    - A tuple containing hours, minutes, seconds, and milliseconds.
    """
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = (total_seconds - int(total_seconds)) * 1000

    return (int(hours), int(minutes), int(seconds), int(milliseconds))


def get_cpu_processes():

    # for local machines
    if (platform.uname().node == 'macbook') or (platform.uname().system == 'Darwin') or (platform.uname().system == 'Windows'):
        cores = int(multiprocessing.cpu_count()/4)

    else: # meant for supercomputers
        cores = int(multiprocessing.cpu_count())

    return cores


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
    if type.lower() == 'natural':
        land_tiff_natural = load_geotiff(os.path.join(parent_dir, 'data/shapefiles/natural_earth_data/10m_natural_earth_relief/NE1_HR_LC_SR.tif'))
        return land_tiff_natural

    elif (type.lower() == 'topo') or (type.lower() == 'hypso'):
        land_tiff_hypsometric = load_geotiff(os.path.join(parent_dir, 'data/shapefiles/natural_earth_data/10m_natural_earth_relief_hypsometric/HYP_HR_SR.tif'))
        return land_tiff_hypsometric

    else: #TODO
        print('Message [load_land_feature]: `type` must be one of hypso, topo, or natural')
