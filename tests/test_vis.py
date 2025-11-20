import shutil
import pytest
import pyemu
import pandas as pd
from pathlib import Path

import pypestvis as ppv

def spinup_freyberg(tmp):
    m_d = Path("examples", "freyberg_ies")
    shutil.copytree(m_d, tmp / m_d.name)
    m_d = tmp / m_d.name
    pst = pyemu.Pst(str(m_d / "freyberg.pst"))
    obs = pst.observation_data
    obs.loc[obs.oname == 'hds', ['k', 'i', 'j']] = obs.loc[obs.oname == 'hds'].obgnme.str.rsplit("_", expand=True, n=3)[
        [1, 2, 3]].values
    pst.observation_data = obs
    return pst

# test currently relying on presence of constructed interface pst_template
def test_freyberg(tmp_path):
    """
    Test the visualization utilities for freyberg.
    """
    from plotly import callbacks
    import numpy as np
    def _check_selval():
        z = vh.map_widget.data[0].z[sel]
        mapfignames = [x.name for x in vh.map_histogram.data]
        selline = vh.map_histogram.data[mapfignames.index('mapval')]
        assert np.unique(selline.x).shape[0] == 1
        selval = selline.x[0]
        assert np.isclose(z, selval), f"histo value ({selval}) does not match map value ({z})"
        return selval

    sel = 10
    m_d = tmp_path / "freyberg_ies"
    pst = spinup_freyberg(tmp_path)
    vh = ppv.VisHandler(pst, wd=m_d)
    z = vh.map_widget.data[0].z[sel]
    vh.map_temporal_slider.value = 1
    z2 = vh.map_widget.data[0].z[sel]
    assert z != z2 # should be different at time 1
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[sel]), None)
    selval = _check_selval()
    assert np.isclose(selval, z2) # should be same after click

    vh.prob_slider.value = 90
    selval2 = _check_selval()
    assert selval2 > selval # should be larger at P90

    vh.reals_or_ptile_radio.value = 'r'
    selval3 = _check_selval()
    assert selval3 != selval2 # should have changed
    vh.real_selector.value = '0'
    selval4 = _check_selval()
    assert selval4 != selval3 # should have changed
    vh.reals_or_ptile_radio.value = 'p'
    selval5 = _check_selval()
    assert selval5 != selval4 # should have changed
    assert selval5 == selval2  # should have changed


def test_nounmap(tmp_path):
    from plotly import callbacks
    m_d = tmp_path / "freyberg_ies"
    pst = spinup_freyberg(tmp_path)
    obs = pst.observation_data
    # strip out unmapable
    obs = obs.loc[obs.oname!='sfr', :]
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    vh.map_temporal_slider.value = 1


def test_nomap(tmp_path):
    from plotly import callbacks
    m_d = tmp_path / "freyberg_ies"
    pst = spinup_freyberg(tmp_path)
    obs = pst.observation_data
    # strip out mapable
    obs = obs.loc[obs.oname=='sfr', :]
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    vh.unmap_temporal_slider.value = 1


def test_no_t(tmp_path):
    from plotly import callbacks
    m_d = tmp_path / "freyberg_ies"
    pst = spinup_freyberg(tmp_path)
    obs = pst.observation_data
    obs = obs.loc[(obs.kper == '0') |
                  ((obs.oname=='sfr') &
                   (obs.time=='1')), :]
    obs['kper'] = obs.kper.fillna(0).astype(int)

    obs = obs.drop(columns=['time'])
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[10]), None)

    obs = obs.drop(columns=['kper', 'kstp'], errors='ignore')
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[10]), None)


def test_t_str(tmp_path):
    from plotly import callbacks
    m_d = tmp_path / "freyberg_ies"
    pst = spinup_freyberg(tmp_path)
    obs = pst.observation_data
    obs = obs.loc[(obs.kper == '0') |
                  ((obs.oname=='sfr') &
                   (obs.time=='1')), :]
    obs['time'].iloc[0] = 'one'
    # should now fail converting to int and set all tslider to str
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    with pytest.raises(IndexError):
        # should not be able to click b\c default map not avail for default tslider value
        vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[10]), None)
    vh.map_temporal_slider.value = 1
    # now map should be avail
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[10]), None)


@pytest.mark.parametrize("option", ['model', 'grb', 'pkl'])
def test_lh(tmp_path, option):
    """
    Test the visualization utilities in pyemu.
    """
    m_d = Path("examples", "lheg_ies")
    shutil.copytree(m_d, tmp_path/m_d.name)
    m_d = tmp_path/m_d.name
    if option != 'model':
        for f in m_d.glob('*.nam'):
            f.unlink()
        if option == 'pkl':
            for f in m_d.glob('*.grb'):
                f.unlink()
    pst = pyemu.Pst(str(m_d / "lhgzsi.pst"))
    obs = pst.observation_data
    scenmap = pd.read_csv(Path(m_d, "scenario.csv")).set_index('kper')
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
    vh = ppv.VisHandler(pst, wd=m_d, crs='EPSG:2913')
    vh._cell_sel_id = 5408
    vh.highlight_cell()
    vh.update_maphisto()
    vh.unmap_group_selector.value = vh.unmap_group_selector.options[vh.unmap_group_selector.index + 1]
    vh.unmap_group_selector.value = vh.unmap_group_selector.options[vh.unmap_group_selector.index + 1]
    vh.weighted_obs_checkbox.value = not vh.weighted_obs_checkbox.value
    vh.set_map()


if __name__ == '__main__':
    # test_vis('test')
    pass