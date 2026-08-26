#!/usr/bin/env python3
import sys
import time
import faulthandler

def p(msg):
    print(msg, flush=True)

faulthandler.enable(all_threads=True)
p('faulthandler enabled')

modules = [
    ('sys', 'sys'),
    ('platform', 'platform'),
    ('os', 'os'),
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
    ('set matplotlib backend', None),
    ('matplotlib.pyplot', 'matplotlib.pyplot'),
    ('cartopy', 'cartopy'),
    ('cartopy.crs', 'cartopy.crs'),
    ('pyproj', 'pyproj'),
    ('rasterio', 'rasterio'),
    ('PIL.Image', 'PIL.Image'),
    ('joblib', 'joblib'),
    ('shapely', 'shapely'),
    ('pyhdf.SD', 'pyhdf.SD'),
]

for name, mod in modules:
    try:
        p(f"importing {name} ...")
        if mod is None:
            import matplotlib
            # force non-interactive backend on headless
            try:
                import platform as _pl
                if not ((_pl.uname().node == 'macbook') or (_pl.uname().system == 'Darwin') or (_pl.uname().system == 'Windows')):
                    matplotlib.use('Agg')
                    p('matplotlib backend set to Agg')
            except Exception as e:
                p(f'warning setting backend: {e}')
        else:
            __import__(mod)
        p(f"import {name} OK")
    except Exception as e:
        p(f"import {name} FAILED: {e}")
    time.sleep(0.2)

p('done')
