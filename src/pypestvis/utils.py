"""
Utils for vis builders
"""

import numpy as np
from pathlib import Path
import pickle
import warnings


def _sort_key(x):
    try:
        return (1, float(x))
    except ValueError:
        return (0, x)


def get_cellid_fromij(idxs, shape):
    # todo: this could be fleshed out to be more flexible
    return np.ravel_multi_index(idxs, shape)


def mg2geojson(mg, crs=None):
    """
    Convert model grid to GeoJSON format.

    This function reads the model grid from the current working directory
    and saves it as a GeoJSON file named 'model_grid.json'.
    """
    lcrs = 'epsg:4326' # WGS84
    from pathlib import Path
    import geopandas as gpd
    from shapely import polygons
    import pandas as pd
    import json

    if crs is None:
        crs = mg.crs
        if crs is None:
            warnings.warn("No crs passed, geojson will be in unprojected coords and may not map correctly")

    if isinstance(mg, (Path, str)):
        mg = get_mg_from_grb(mg)
    ib = mg.idomain.reshape(mg.shape)
    # Create a GeoDataFrame from the model grid
    # we are going to try and just create a single layer json
    # this might work for struct and maybe disv bu disu will need something else.
    # for now just using ij to build grid
    cells = pd.DataFrame(np.argwhere(ib.any(axis=0)), columns=['i', 'j'])
    cells['in_verts'] = polygons(np.array(
        mg.get_cell_vertices(cells.i.values, cells.j.values)  # uses baked in flopy method
        ).transpose((2, 0, 1)).tolist())
    cells['cellid'] = get_cellid_fromij(tuple(cells[['i','j']].values.T), mg.shape[1:])

    # cells = pd.DataFrame(np.argwhere(ib != 0), columns=['k', 'i', 'j'])
    # cells['in_verts'] = polygons(np.array(
    #     mg.get_cell_vertices(cells.i.values, cells.j.values)  # uses baked in flopy method
    #     ).transpose((2, 0, 1)).tolist())
    # cells['cellid'] = mg.get_node(cells[['k','i','j']].values.tolist())
    geoms = gpd.GeoSeries(cells['in_verts'], crs=crs)
    if crs is not None: # project to lat/lon
        geoms = geoms.to_crs(lcrs)
    cells = gpd.GeoDataFrame(cells, geometry=geoms)
    cells = cells.set_index('cellid').geometry
    asjson = cells.to_json(show_bbox=False)
    return json.loads(asjson)


def get_geojson(geojson, mg=None, crs=None, wd=None,
                write=False):
    import json
    assert any([geojson, mg, wd]), "one of geojson, mg, or wd must be provided"
    _mg = mg
    if isinstance(geojson, (str, Path)):
        # will be Path as default to saving in assets
        try:
            with open(geojson, 'r') as fp:
                geojson = json.load(fp)
            return geojson  # will return without writing (all good)
        except FileNotFoundError:
            pass
        if mg is None:
            _mg, _ = get_mg_mt(wd)
        # and can be used to identify to map data to the grid (e.g a cellid)
        _geojson = mg2geojson(_mg, crs=crs)
        if write:
            Path(geojson).parent.mkdir(parents=True, exist_ok=True)
            with open(geojson, 'w') as fp:
                json.dump(_geojson, fp)
        geojson = _geojson
    return geojson


def get_mg_mt(d):
    """
    Get model grid and model time from model working directory.

    Parameters
    ----------
    d : str or Path
        Path to model working directory.

    Returns
    -------
    mg : flopy.ModelGrid
        Model grid object.
    mt : flopy.ModelTime
    """
    import flopy
    from pathlib import Path
    # TODO: more flexiblity around modelgrid and time definitions -- less reliance on mf6/flopy and unstruct options?
    d = Path(d)
    try:
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=d,
            version='mf6',
            # exe_name='mf6',
            verbosity_level=0,
            load_only=['dis', 'tdis'],
            lazy_io=True
        )
        gwf = sim.get_model()
        mg = gwf.modelgrid
        mt = gwf.modeltime
    except Exception as e:
        try:
            mg = get_mg_from_grb(d)
        except Exception:
            with open(d / 'modelgrid.pkl', 'rb') as f:
                mg = pickle.load(f)
        try:
            with open(d / 'modeltime.pkl', 'rb') as f:
                mt = pickle.load(f)
        except FileNotFoundError:
            mt = None   # todo: more flexiblity downstream when there is not modeltime info
            warnings.warn("Can't load model time object")
    return mg, mt


def get_mg_from_grb(wd):
    import flopy
    grb = list(wd.glob('*.grb'))
    if len(grb) == 0:
        raise FileNotFoundError("No grb files found")
    mg = flopy.mf6.utils.MfGrdFile(grb[0]).modelgrid
    return mg

def _check_gridmappable(coords, min_size=20):
    """Return True if any cluster of adjacent (i, j) has at least min_size members."""
    from collections import deque

    coords_set = set(map(tuple, coords))
    visited = set()
    max_cluster = 0

    for cell in coords_set:
        if cell in visited:
            continue
        # BFS for this cluster
        queue = deque([cell])
        cluster = set([cell])
        while queue:
            ci, cj = queue.popleft()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (ci + di, cj + dj)
                if neighbor in coords_set and neighbor not in cluster:
                    cluster.add(neighbor)
                    queue.append(neighbor)
        visited |= cluster
        max_cluster = max(max_cluster, len(cluster))
        if max_cluster >= min_size:
            return True
    return False

def _guess_mappable(df):
    """Guess if the observation group is mappable based on its structure."""
    if df.i.notna().all():
        # mappable but might not be gridded
        # makesure k, i, j are integers
        if _check_gridmappable(df[['i', 'j']].values):
            return 'grid'
        else:
            return 'point'
    else:
        return 'unmap'


def _nat_sort(listlike):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
    return sorted(listlike, key=alphanum_key)
