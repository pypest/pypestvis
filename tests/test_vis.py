import pytest
import pyemu
import pandas as pd
from pathlib import Path

import pypestvis as ppv

# test currently relying on presence of constructed interface pst_template
def test_freyberg():
    """
    Test the visualization utilities for freyberg.
    """
    m_d = Path('..', "examples", "freyberg_ies")
    pst = pyemu.Pst(str(m_d / "freyberg.pst"))
    obs = pst.observation_data
    obs.loc[obs.oname=='hds', ['k', 'i', 'j']] = obs.loc[obs.oname=='hds'].obgnme.str.rsplit("_",expand=True, n=3)[[1,2,3]].values
    pst.observation_data = obs
    vh = ppv.VisHandler(pst, wd=m_d)
    pass

def test_lh():
    """
    Test the visualization utilities in pyemu.
    """
    m_d = Path('..', "examples", "lheg_ies")
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
    vh = ppv.VisHandler(pst, wd=m_d)
    vh._cell_sel_id = 5408
    vh.highlight_cell()
    vh.update_maphisto()
    vh.unmapgroupselector.value = vh.unmapgroupselector.options[vh.unmapgroupselector.index+1]
    vh.unmapgroupselector.value = vh.unmapgroupselector.options[vh.unmapgroupselector.index+1]
    vh.wobselector.value = not vh.wobselector.value
    vh.set_map()


if __name__ == '__main__':
    # test_vis('test')
    pass