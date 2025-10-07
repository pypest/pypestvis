"""
Utils for vis builders
"""

import pyemu
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import warnings


def _sort_key(x):
    try:
        return (1, float(x))
    except ValueError:
        return (0, x)


def mg2geojson(mg, wd=None, crs='epsg:2193'):
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

    if isinstance(mg, (Path, str)):
        mg = get_mg_from_grb(mg)
    ib = mg.idomain.reshape(mg.shape)

    # Create a GeoDataFrame from the model grid
    cells = pd.DataFrame(np.argwhere(ib != 0), columns=['k', 'i', 'j'])
    cells['in_verts'] = polygons(np.array(
        mg.get_cell_vertices(cells.i.values, cells.j.values)  # uses baked in flopy method
        ).transpose((2, 0, 1)).tolist())
    cells['cellid'] = mg.get_node(cells[['k','i','j']].values.tolist())
    cells = gpd.GeoDataFrame(cells, geometry=gpd.GeoSeries(cells['in_verts'], crs=crs).to_crs(lcrs))
    cells = cells.drop(columns=['in_verts']).set_index('cellid')
    if wd is not None:
        cells.to_file(Path(wd, f'model_grid.json'), driver='GeoJSON')
    return json.loads(cells.to_json())


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

    d = Path(d)
    try:
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=d,
            version='mf6',
            # exe_name='mf6',
            verbosity_level=0,
            load_only=['dis', 'tdis', 'grb'],
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
            mt = None
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


if __name__ == "__main__":
    from .core import VisHandler
    # from IPython.display import display
    md = Path("../..", "master_median_scen")
    pst=pyemu.Pst(Path(md,"lhgzsi.pst").as_posix())
    obs = pst.observation_data
    scenmap = pd.read_csv(Path(md, "scenario.csv")).set_index('kper')
    chdmap = scenmap.CHD.fillna('none').to_dict()
    # fix up i,j,ks to make zero based
    # obs.groupby('obgnme').first()
    # clean up metadata
    # fill kper etc
    obs[['kper', 'kstp']] = obs[['kper', 'kstp']].astype('Int32').fillna(0)
    # fixing issue with dummy obs that made it incompatible
    obs['idx0'] = obs.idx0.replace('dummy', None)

    # in this instance safe to fill k,i,j with idx0...
    obs = obs.fillna({'k': obs.idx0, 'i': obs.idx1, 'j': obs.idx2}).astype(
        {c: "Int32" for c in ['k', 'i', 'j', 'kstp', 'kper']}).fillna({'k': 0})

    # need annoying one-based (parfile tables) to zerobased
    obs.loc[obs.obgnme.str.contains("chd|ghb", na=False), ['k', 'i', 'j']] -= 1
    obs['slider'] = obs.kper.apply(lambda x: (x, chdmap[int(x)]))
    # put back on pest object
    pst.observation_data = obs
    # vh = VisHandler(pst, wd=md, groupby='obgnme')
    # display(vh.wobselector)
    # vh.wobselector.value = not vh.wobselector.value
    # vh.set_slider_options(index_col='slider')
    vh = VisHandler(pst, wd=md, groupby='obgnme', tidx='slider')
    vh._cell_sel_id = 5408
    vh.highlight_cell()
    vh.update_maphisto()
    vh.mapselector.value = vh.mapselector.options[3]
    vh.mapselector.value = vh.mapselector.options[1]
    vh.wobselector.value = not vh.wobselector.value
    vh.set_map()
    pass
