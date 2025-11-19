import shutil
import pytest
from pypestvis.utils import *


@pytest.mark.parametrize("md", ['lheg_ies', 'freyberg_ies'])
@pytest.mark.parametrize("option", ['model', 'grb', 'pkl'])
def test_mgmt(tmp_path, option, md):
    if md == 'freyberg_ies' and option != 'model':
        pytest.skip("freyberg grb/pkl test skipped-- no grb/pkl files")
    m_d = Path("examples", md)
    shutil.copytree(m_d, tmp_path/m_d.name)
    m_d = tmp_path/m_d.name
    if option != 'model':
        # remove nam files so model load fails
        for f in m_d.glob('*.nam'):
            f.unlink()
        if option == 'pkl':
            # remove grb files so grb load fails
            for f in m_d.glob('*.grb'):
                f.unlink()
    mg, mt = get_mg_mt(m_d)


@pytest.mark.parametrize("crs", [None, True])
@pytest.mark.parametrize("md", ['lheg_ies', 'freyberg_ies'])
def test_json(tmp_path, md, crs):
    m_d = Path("examples", md)
    shutil.copytree(m_d, tmp_path / m_d.name)
    m_d = tmp_path / m_d.name
    if crs:
        if md == 'freyberg_ies':
            crs = "epsg:32614"
        else:
            crs = "epsg:2913"
    # first from fallback path:
    gjsn = get_geojson(wd=m_d, crs=crs)
    # this should have written a json file
    gjsn2 = get_geojson(geojson=m_d / 'model_grid.json')
    assert gjsn == gjsn2
    # now passing gjsn directly
    gjsn3 = get_geojson(geojson=gjsn)
    assert gjsn3 == gjsn2



