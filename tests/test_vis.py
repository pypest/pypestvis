import shutil
import pytest
import pyemu
import pandas as pd
from pathlib import Path
import numpy as np

import pypestvis as ppv

def spinup_freyberg(tmp):
    m_d = Path("examples", "freyberg_ies")
    shutil.copytree(m_d, tmp / m_d.name)
    m_d = tmp / m_d.name
    pst = pyemu.Pst(str(m_d / "freyberg.pst"))
    obs = pst.observation_data
    obs.loc[obs.oname == 'hds', ['k', 'i', 'j']] = \
        obs.loc[obs.oname == 'hds'].obgnme.str.rsplit(
            "_", expand=True, n=3)[[1, 2, 3]].values
    # split hdar into a second group
    obs.loc[(obs.obgnme == 'hdar') & (obs.i.astype("Int32")>20), 'obgnme'] = 'hdar2'
    othersel = ~obs.obgnme.str.startswith('hdar')
    obs.loc[othersel, 'obgnme'] = obs.loc[othersel, 'usecol']
    pst.observation_data = obs
    return pst


def _check_selval(vh, sel=10):
    z = vh.map_widget.data[0].z[sel]
    mapfignames = [x.name for x in vh.map_histogram.data]
    selline = vh.map_histogram.data[mapfignames.index('mapval')]
    assert np.unique(selline.x).shape[0] == 1
    selval = selline.x[0]
    assert np.isclose(z, selval), f"histo value ({selval}) does not match map value ({z})"
    return selval


# test currently relying on presence of constructed interface pst_template
def test_freyberg(tmp_path):
    """
    Test the visualization utilities for freyberg.
    """
    from plotly import callbacks
    import numpy as np
    sel = 10
    m_d = tmp_path / "freyberg_ies"
    pst = spinup_freyberg(tmp_path)
    vh = ppv.VisHandler(pst, wd=m_d, crs="epsg:32614")
    z = vh.map_widget.data[0].z[sel]
    vh.map_temporal_slider.value = 1
    z2 = vh.map_widget.data[0].z[sel]
    assert z != z2 # should be different at time 1
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[sel]), None)
    selval = _check_selval(vh, sel)
    assert np.isclose(selval, z2) # should be same after click

    # switch obgnme
    vh.map_obs_selector.value = 'hdar2'
    assert len(vh.map_histogram.data[0].x) == 0  # should have cleared histo
    vh.map_obs_selector.value = 'hdar'

    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[sel]), None)
    vh.prob_slider.value = 90
    selval2 = _check_selval(vh, sel)
    assert selval2 > selval # should be larger at P90

    vh.reals_or_ptile_radio.value = 'r'
    selval3 = _check_selval(vh, sel)
    assert selval3 != selval2 # should have changed
    vh.real_selector.value = '0'
    selval4 = _check_selval(vh, sel)
    assert selval4 != selval3 # should have changed
    vh.reals_or_ptile_radio.value = 'p'
    selval5 = _check_selval(vh, sel)
    assert selval5 != selval4 # should have changed
    assert selval5 == selval2  # should have changed

    vh.map_obs_selector.value = 'trgw'
    vh.on_map_click(vh.map_widget.data[1], callbacks.Points(point_inds=[0]), None)

    pass


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
    # expecting user to provide a kper col if no time col
    # (and some mapping to time -- e.g. mt)
    obs['kper'] = obs.kper.fillna(0).astype(int)
    obs = obs.drop(columns=['time'])
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    # test a click callback
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[10]), None)

    # test with no kper either
    obs = obs.drop(columns=['kper', 'kstp'], errors='ignore')
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[10]), None)


def test_t_str(tmp_path):
    # testing (some) strings in time col
    from plotly import callbacks
    m_d = tmp_path / "freyberg_ies"
    pst = spinup_freyberg(tmp_path)
    obs = pst.observation_data
    obs = obs.loc[(obs.kper == '0') |
                  ((obs.oname=='sfr') &
                   (obs.time=='1')), :]
    obs['time'].iloc[0] = 'one'
    obs['time'].iloc[3] = 'two'
    pst.observation_data = obs
    sel = 10
    vh = ppv.VisHandler(pst, wd=m_d)
    # value at selection
    z = vh.map_widget.data[0].z[sel]
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[sel]), None)
    selval = _check_selval(vh, sel)
    # default k should be 0 and only 1 time
    assert 1 not in [i for v,i in vh.map_temporal_slider.options]

    # change layer
    vh.layer_selector.value = '1'
    # sel should be diff as active locations different in layer 1
    sel2 = np.where(vh.map_widget.data[0].locations == vh._sel_cellid)[0][0]
    selval = _check_selval(vh, sel2)
    z2 = vh.map_widget.data[0].z[sel2]
    # new values should be diff to previous layer
    assert z2 != z

    # temporal slider to 1 -- should be available in layer 2
    vh.map_temporal_slider.value = 1
    # should only be one value now
    assert len(vh.map_widget.data[0].z) == 1
    z3 = vh.map_widget.data[0].z[0]
    assert z3 != z2
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[0]), None)
    selval = _check_selval(vh, 0)
    assert selval == z3

    # back to top layer
    vh.layer_selector.value = '0'
    # back to original time
    vh.map_temporal_slider.value = 0
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[sel]), None)
    selval = _check_selval(vh, sel)
    # val to should be equiv. to original
    assert selval == z


def test_weighted_check(tmp_path):
    from plotly import callbacks
    from traitlets import TraitError
    m_d = Path("examples", "freyberg_ies")
    pst = spinup_freyberg(tmp_path)
    obs = pst.observation_data
    obs.loc[obs.obgnme=='hdar', 'weight'] = 0
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    assert vh.weighted_obs_checkbox.value is False
    assert vh.map_obs_selector.value == 'hdar'
    vh.on_map_click(vh.map_widget.data[0],
                    callbacks.Points(point_inds=[10]), None)
    selval = _check_selval(vh, 10)
    vh.weighted_obs_checkbox.value = True
    assert vh.map_obs_selector.value == 'hdar2'  # should have switched
    assert vh._sel_cellid is None
    try:
        vh.map_obs_selector.value = 'hdar'
    except TraitError:
        pass  # should not be able to set back to unweighted
    else:
        raise Exception("should not be able to set obs selector back to unweighted")
    vh.weighted_obs_checkbox.value = False
    assert vh.map_obs_selector.value == 'hdar2'  # should still be the same
    vh.map_obs_selector.value = 'hdar'
    vh.on_map_click(vh.map_widget.data[0],
                    callbacks.Points(point_inds=[10]), None)
    selval2 = _check_selval(vh, 10)
    assert selval2 == selval  # should be same


def test_no_weighted(tmp_path):
    from plotly import callbacks
    m_d = Path("examples", "freyberg_ies")
    pst = spinup_freyberg(tmp_path)
    obs = pst.observation_data
    obs['weight'] = 0
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    assert vh.weighted_obs_checkbox.value is False
    assert vh.weighted_obs_checkbox.disabled is True
    vh.on_map_click(vh.map_widget.data[0],
                    callbacks.Points(point_inds=[10]), None)
    selval = _check_selval(vh, 10)
    obsdatas = [d for d in vh.map_histogram.data if 'obs' in d.name]
    for d in obsdatas:
        assert not np.any(d.x), \
            f"obs should be empty for zero weights, check {d.name}"
    # try force weighted
    vh.weighted_obs_checkbox.value = True
    # should be rejected
    assert vh.weighted_obs_checkbox.value is False


def test_no_obsplus(tmp_path):
    from plotly import callbacks
    m_d = Path("examples", "freyberg_ies")
    pst = spinup_freyberg(tmp_path)
    m_d = tmp_path / m_d.name
    for d in m_d.glob("*obs+noise*"):
        d.unlink()
    vh = ppv.VisHandler(pst, wd=m_d)
    vh.on_map_click(vh.map_widget.data[0],
                    callbacks.Points(point_inds=[10]), None)
    selval = _check_selval(vh, 10)
    obsdatas = [d for d in vh.map_histogram.data if 'obs+plus' in d.name]
    for d in obsdatas:
        assert not np.any(d.x), \
            f"obs+noise should be empty for zero weights, check {d.name}"


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
    pst = str(m_d / "lhgzsi.pst")
    vh = ppv.VisHandler(pst, wd=m_d, crs='EPSG:2913')
    vh._sel_cellid = 5408
    vh.highlight_cell()
    vh.update_maphisto()
    vh.unmap_group_selector.value = vh.unmap_group_selector.options[vh.unmap_group_selector.index + 1]
    vh.unmap_group_selector.value = vh.unmap_group_selector.options[vh.unmap_group_selector.index + 1]
    vh.weighted_obs_checkbox.value = not vh.weighted_obs_checkbox.value
    vh.set_map()


def profile_vis(wd=None, temp=Path('profiling'), crs=None):
    import cProfile
    import pstats
    from plotly import callbacks

    if wd is None:
        wd = 'freyberg_ies'
        m_d = Path(temp, wd)
        shutil.rmtree(m_d, ignore_errors=True)
        pst = spinup_freyberg(temp)
    else:
        wd = Path(wd)
        m_d = Path(temp, wd.name)
        shutil.copytree(wd, m_d, dirs_exist_ok=True)
        pstfname = list(m_d.glob('*.pst'))[0]
        pst = pyemu.Pst(str(pstfname))
    Path("assets", f"{Path(pst.filename).stem}_modelgrid.json").unlink(missing_ok=True)
    pr = cProfile.Profile()
    pr.enable()
    vh = ppv.VisHandler(pst, wd=m_d, crs=crs, write_json=True)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(40)

    pr = cProfile.Profile()
    pr.enable()
    vh.on_map_click(vh.map_widget.data[0], callbacks.Points(point_inds=[10]), None)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(40)

    assert Path("assets", f"{vh.name}_modelgrid.json").exists()
    pr = cProfile.Profile()
    pr.enable()
    vh = ppv.VisHandler(pst, wd=m_d, crs=crs)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(40)

    # pr = cProfile.Profile()
    # pr.enable()
    # vh.

if __name__ == '__main__':
    # test_vis('test')
    # profile_vis(crs="epsg:32614")
    # profile_vis(crs="epsg:32614")
    alt = dict(wd=Path("..", "..", "ranger_ua", "master_precond"),
               crs="EPSG:28353")
    profile_vis(**alt)
    pass