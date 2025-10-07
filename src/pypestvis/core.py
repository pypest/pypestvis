"""Core functionality of pypestvis"""

__all__ = ["VisHandler", "VisGroupHandler"]

import pandas as pd
import plotly.colors as pc
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pyemu
import ipywidgets as ipyw
from plotly import graph_objects as go
import warnings
from shapely.geometry import shape

from .utils import _guess_mappable, get_mg_mt, mg2geojson, _sort_key


class VisGroupHandler(object):
    """
    Handler Class for groups in the web application.
    """
    def __init__(self, df, mg, ens=None, tidx='time'):
        """
        Parameters
        ----------
        df : pd.DataFrame
        mg : flopy.ModelGrid
            Used for getting cellid or x,y from k,i,j metadata stored in df
        ens : pd.DataFrame, optional
            Ensemble indexed by obs/par names with columns as multiindex of (iterations, realization).
        tidx : str, optional
            A column in observation dataframe to use as temporal indexer for slider widget selection.
            Default is 'time'. which will be inferred from kper, kstp if absent and mt is passed.
        """
        self.mapable = _guess_mappable(df)
        if self.mapable == 'grid':
            df['cellid'] = mg.get_node(df[['k','i','j']].values.tolist())
        if self.mapable == 'point':
            if 'x' not in df.columns:
                df['x'] = mg.xcellcenters[df.i.values, df.j.values]
            if 'y' not in df.columns:
                df['y'] = mg.ycellcenters[df.i.values, df.j.values]
            df = df.fillna({'x': pd.Series(mg.xcellcenters[df.i.values, df.j.values]),
                           'y': pd.Series(mg.ycellcenters[df.i.values, df.j.values])})

        self.metadf = df.copy()
        if self.mapable == 'grid':
            idxcols = ['cellid', 'k', tidx]
        elif self.mapable == 'point':
            idxcols = ['x', 'y', 'k', tidx]
        else:
            idxcols = ['usecol', tidx]
        idxname = df.index.name
        self.idxmap = df.loc[:, idxcols]
        self.idxmap_r = self.idxmap.reset_index().groupby(idxcols)[idxname].unique()

        if ens is None:
            self.ens = None
            self.qtiles = None
        else:
            gpens = ens.loc[df.index, :].copy()
            gpens.index = pd.MultiIndex.from_frame(df[idxcols])
            self.ens = gpens
            self.qtiles = self.ens.T.groupby(level='iteration').quantile(
                np.linspace(start=0, stop=1, num=21)
            ).T
            # rename percentiles
            self.qtiles.columns = self.qtiles.columns = self.qtiles.columns.set_levels(
                self.qtiles.columns.levels[1].map(lambda x: f"P{int(100 * x)}"),
                level=1
            )


class VisHandler(object):
    """
    Handler for visualizations in the web application. Currently flopy mf6 modelgrid dependent.
    """
    def __init__(self,
                 pst,
                 geojson=None,  # needed for mapping, could add additional geojson options as extra kwargs?
                 wd=None,  # working directory for the model, needed to get mg from grb, also a save location
                 mg=None,  # needed for referencing kij to json, also can be used to build json if geojson is absent
                 mt=None,  # model time, needed for obs data
                 crs='epsg:2193',  # coordinate reference system for the modelgrid -- will be converted to WGS84
                 groupby='obgnme',  # groupby for the obs data, default is obgnme
                 tidx='time'):
        """

        Parameters
        ----------
        pst : pyemu.Pst or str or Path
        geojson : str or dict, optional
        wd : str or Path, optional
        mg : flopy.ModelGrid, optional
        mt : flopy.ModelTime, optional
        crs : str, optional
            Optional coordinate reference system for the model grid. Only used if geojson is None when it is passed to
            constructor method `mg2geojson` to build json from modelgrid object.
            Default is 'epsg:2193'.
        groupby : str, optional
        tidx: str, optional
        """
        self._callback_off = False
        if isinstance(pst, (str, Path)):
            pst = pyemu.Pst(str(pst))
        self.pst = pst

        _mg = mg
        _mt = mt
        if (mg is None) or (mt is None):
            assert wd is not None, "Must provide wd if either mg and mt as None"
            _mg, _mt = get_mg_mt(wd)
            if mg is None:
                mg = _mg
            if mt is None:
                mt = _mt
        self.mg = mg
        self.mt = mt
        self.tidx = tidx

        if geojson is None: # need a geojson for mapping -- it also need to have a property that is unigue
                            # and can be used to identify to map data to the grid (e.g a cellid)
            geojson = mg2geojson(mg, crs=crs)
        if isinstance(pst, (str, Path)):
            with open(geojson, 'r') as fp:
                geojson = geojson.load(fp)
        self.geojson = geojson

        # lists for storing tags of mappable status of data groups
        self.gridmapable = []
        self.pointmapable = []
        self.unmapable = []
        self.weighted = []

        self.groupby = groupby

        self.obs_dict = {}
        self.obsval_dict = {}
        self.par_dict = {}
        self.real_dict = {}
        self._build_obs_handlers()
        self._cell_sel_id = None
        self._uservminmax = False # for storing if user has set vmin/vmax
        self._set_widgets()

    @contextmanager
    def callback_off(self):
        self._callback_off = True
        try:
            yield
        finally:
            self._callback_off = False

    def _set_widgets(self):
        gridw = self.obsval_dict.keys() & self.gridmapable
        self.wobselector = ipyw.Checkbox(
            value=False,
            description="Weighted only",
            disabled=False if len(gridw) > 0 else True,
        )

        self.mapselector = ipyw.RadioButtons(
            options=self.gridmapable,
            # value=self.gridmapable[0],
            description='Gridded datasets:',
            disabled=False
        )
        self.vminmaxslider = ipyw.FloatRangeSlider(
            value=[-1e30, 1e30],
            min=-1e30,
            max=1e30,
            step=100,
            description='Colorscale range:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
        )
        self.vminmaxbutton = ipyw.Button(
            description='Reset range',
            disabled=False,
            button_style='',
            tooltip='Rest vmin/vmax to data range',
            icon="arrows-left-right-to-line"
        )
        self.vminmaxbutton.on_click(self._reset_vminmax)

        self.vminvmax = ipyw.VBox([self.vminmaxslider, self.vminmaxbutton])

        self.layselector = ipyw.Dropdown(
            options=[],
            value=None,
            description='Layer:',
            disabled=False,
        )

        self.rpselector = ipyw.RadioButtons(
            options=[('Select reals.', 'r'), ("Select P.", 'p')],
            value='p',
            description='Plot type:',
            disabled=False,
        )

        self.pslider = ipyw.FloatSlider(
            value=50,
            min=0,
            max=100,
            step=5,
            description='Percentile:',
            disabled=False,
            continuous_update=False,
            orientation='vertical',
            readout=True,
            readout_format='.0f',
        )

        self.tslider = ipyw.SelectionSlider(
            options=[()],
            # value=None,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            # readout_format='.0f',
        )
        self.set_slider_options(index_col=self.tidx)

        self.iterselector = ipyw.Dropdown(
            options=zip(*[sorted(self.real_dict.keys())]*2),
            description="Iteration: ",
            value=sorted(self.real_dict.keys())[0]
        )
        self.realselector = ipyw.Dropdown(options=sorted(self.real_dict[self.iterselector.value].tolist(), key=_sort_key),
                                          description="Realisation: ",
                                          disabled=True,)

        self.cmapselector = ipyw.Dropdown(
            options=pc.named_colorscales(),
            value='Plasma'.lower(),
            description='Colorscale:',
        )
        self.cmapreverse = ipyw.Checkbox(
            value=False,
            description="Reverse",
            disabled=False,
            layout={"align_self": "flex-start"}
        )

        self.logselector = ipyw.Checkbox(value=False,
                                         description="Logscale",
                                         disabled=False)

        self.rpselector.observe(self.rpchange, names=['value'])

        self.mapselector.observe(self.set_bounds_and_map, names=['value'])

        self.pslider.observe(self.set_map, names=['value'])
        self.layselector.observe(self.set_map, names=['value'])
        self.iterselector.observe(self.set_map, names=['value'])
        self.realselector.observe(self.set_map, names=['value'])
        self.cmapselector.observe(self.set_map, names=['value'])
        self.cmapreverse.observe(self.set_map, names=['value'])

        self.logselector.observe(self.set_both, names=['value'])
        self.tslider.observe(self.set_both, names=['value'])

        self.wobselector.observe(self.set_mapselector, names=['value'])

        # non mapable widgets
        self.unmaphisto = None
        self.unmapselector = None
        self.unmapgroupselector = None
        if len(self.unmapable) > 0:
            self.get_unmap_widgets()

        self.mapwidget, self.maphisto = self.get_plotly_mapfig() if self.geojson else (None,None)
        self._reset_vminmax()
        self.vminmaxslider.observe(self.set_vminmax, names=['value'])

    def set_vminmax(self, change=None):
        """
        Set the vmin and vmax values for the color scale based on the current map data values
        """
        print('Setting vmin and vmax')
        self._uservminmax = True
        self.set_map(change=change)

    def _reset_vminmax(self, change=None, mapfig=None):
        """
        Set the vmin and vmax values for the color scale based on the current map data values
        """
        if mapfig is None:
            mapfig = self.mapwidget
        self._uservminmax = False
        vmin = mapfig.data[0].z.min()
        vmax = mapfig.data[0].z.max()
        with self.vminmaxslider.hold_trait_notifications():
            self.vminmaxslider.min = vmin
            self.vminmaxslider.max = vmax
            self.vminmaxslider.step = (vmax - vmin) / 1000.0
            self.vminmaxslider.value = [vmin, vmax]

    def _build_obs_handlers(self):
        obs = self.pst.observation_data
        incols = obs.columns.intersection({'kper', 'kstp', 'k', 'i', 'j'})
        obs = obs.astype({c:"Int32" for c in incols})
        if self.mt is not None and self.tidx == 'time':
            # need to get 'time' col
            if 'time' not in obs.columns:
                obs['time'] = np.nan
            # this will need generalising
            if 'kper' in obs.columns:
                if 'kstp' not in obs.columns:
                    obs['time'] = obs.apply(
                        lambda x: self.mt.get_elapsed_time(x.kper if not pd.isna(x.kper) else 0, None)
                        if pd.isna(x.time) else x.time, axis=1
                    ).astype(float)
                else:
                    obs['time'] = obs.apply(
                        lambda x: self.mt.get_elapsed_time(x.kper if not pd.isna(x.kper) else 0, x.kstp) if pd.isna(
                            x.time) else x.time, axis=1).astype(float)

        self.pst.observation_data = obs
        ens = self.pst.ies.obsen.T
        if 'iteration' not in ens.columns.names:
            ens = pd.concat({0: ens}, axis=1, names=['iteration'])
        # handy lookup for realizations for each iteration
        self.real_dict = ens.columns.to_frame(False).groupby('iteration').realization.unique().to_dict()
        for gp, obdf in obs.groupby(self.groupby):
            gph = VisGroupHandler(obdf, mg=self.mg, ens=ens, tidx=self.tidx)
            if gph.mapable == 'grid':
                self.gridmapable.append(gp)
            elif gph.mapable == 'point':
                self.pointmapable.append(gp)
            else:
                self.unmapable.append(gp)
            if any(obdf.weight != 0):
                wobs = obdf.loc[obdf.weight != 0]
                # todo: catch and forgive absent noise ensembles
                self.obsval_dict[gp] = self.pst.ies.noise[wobs.index].T
            self.obs_dict[gp] = gph
        pass

    def _get_tidx(self):
        t = self.tslider.options[self.tslider.index]
        if len(t) > 1:
            return t[0]
        return t

    def get_unmap_widgets(self):
        ig = self.unmapable[0]
        unmapgsel = ipyw.Dropdown(options=self.unmapable,
                                  description="Non-mappable groups: ",
                                  value=ig,
                                  style={'description_width': 'initial'})
        gph = self.obs_dict[ig]
        t = self._get_tidx()
        opts = gph.ens.index.to_frame()
        opts = opts.loc[opts[self.tidx] == t].index.unique(level=0)
        unmapsel = ipyw.Dropdown(options=opts,
                                 description="Non-mappable obs: ",
                                 value=opts[0],
                                 style={'description_width': 'initial'})
        seldf = gph.ens.loc[(opts[0], [t]), :]
        if len(seldf) > 1:
            warnings.warn("output and tidx match more than one output",
                          UserWarning)
        seldf = seldf.iloc[0]

        unmaphisto = go.Figure([go.Histogram(x=seldf.loc[(i, slice(None))].values,
                                                   histnorm='probability density',
                                                   name=f"iter_{i}",
                                                   opacity=0.75) for i in
                                      sorted(self.real_dict.keys())],
                                     layout=dict(barmode='overlay',
                                                 height=400,
                                                 width=600,
                                                 margin=dict(t=10, b=10, l=10, r=10),
                                                 yaxis2=dict(overlaying="y", range=[0,1], visible=False)))
        unmaphisto.add_trace(go.Histogram(
            marker_color='rgba(0,0,0,0)',  # Transparent fill
            marker_line_color='red',  # Outline color
            marker_line_width=2,
            name=f"obs+noise",
            histnorm='probability density',
        ))
        unmaphisto.add_trace(go.Scatter(
            x=[None]*50, y=np.linspace(0,1,50),
            line=dict(color='red', width=3, dash='dash'),
            name='obsval',
            yaxis='y2',
            showlegend=False,
            hovertemplate="obsval: %{x}<extra></extra>",
            visible=False
        ))

        unmaphisto = go.FigureWidget(unmaphisto)


        unmapgsel.observe(self.unmapgchange, names=['value'])
        unmapsel.observe(self.unmapchange, names=['value'])

        self.unmaphisto = unmaphisto
        self.unmapgroupselector = unmapgsel
        self.unmapselector = unmapsel

    def set_mapselector(self, change=None):
        with self.callback_off():
            cv = self.mapselector.value
            if self.wobselector.value:
                gridw = self.obsval_dict.keys() & self.gridmapable
                if len(gridw) > 0:
                    self.mapselector.options = sorted(gridw)
                else:
                    self.wobselector.value = False
                    self.wobselector.disabled = True
            else:
                self.mapselector.options = self.gridmapable
            if cv in self.mapselector.options:
                self.mapselector.value = cv
            else:
                self.mapselector.value = self.mapselector.options[0]
        self.set_map(change)

    def set_layselector(self, gp):
        # get current layer selector value
        k = self.layselector.value
        # get group handler for selected group
        gph = self.obs_dict[gp]
        kopt = sorted(gph.metadf.k.unique().tolist())
        if k is None or k not in kopt:
            self.layselector.options = kopt
            self.layselector.value = kopt[0]
        else:
            self.layselector.options = kopt
            self.layselector.value = k
        return self.layselector.value

    def set_bounds_and_map(self, change=None):
        self._uservminmax = False
        self.set_map(change=change)

    def set_map(self, change=None, mapfig=None):
        if self._callback_off:
            # if we are in a callback, don't do anything
            return
        # will be used in callback so need to handle change arg
        if mapfig is None:
            mapfig = self.mapwidget
        gp = self.mapselector.value
        gph = self.obs_dict[gp]
        i = self.iterselector.value
        with self.callback_off():
            self.set_layselector(gp)
        k = self.layselector.value
        t = self._get_tidx()
        self.realselector.options = sorted(self.real_dict[i].tolist(), key=_sort_key)
        r = self.realselector.value
        p = self.pslider.value
        log = self.logselector.value
        cmap = self.cmapselector.value
        cr = self.cmapreverse.value
        if self.rpselector.value == 'r':
            seldf = gph.ens.loc[(slice(None), k, t), (i, r)]
        else:
            seldf = gph.qtiles.loc[(slice(None), k, t), (i, f"P{int(p)}")]
        if cr:
            cmap += '_r'
        if self.wobselector.value:
            obscells = gph.idxmap.loc[self.obsval_dict[gp].index].cellid.values
            seldf = seldf.loc[(obscells, k, slice(None))]
        z = seldf.values
        # print(seldf)
        if log:
            z = np.log10(z)

        if self._uservminmax:
            zmin, zmax = self.vminmaxslider.value
            zmin = np.max([zmin, z.min()])
            zmax = np.min([zmax, z.max()])
        else:
            zmin, zmax = [None, None]
        print("vminvmax: ", zmin, zmax)
        with mapfig.batch_update():
            mapfig.update_traces(
                z=z,
                zmin=zmin,
                zmax=zmax,
                zauto=True if zmin is None or zmax is None else False,
                locations=seldf.index.get_level_values('cellid'),
                colorscale=cmap,
                customdata=seldf.values,
                selector=dict(name='cpmap')
            )
        if not self._uservminmax:
            with self.callback_off():
                self._reset_vminmax(mapfig=mapfig)
        self.highlight_cell(mapfig)
        if change is not None:
            if change['owner'] == self.realselector or change['owner'] == self.pslider:
                print("Only updating guide line")
                self.update_maphisto_line()
            else:
                print("Updating histogram")
                self.update_maphisto()

    def set_both(self, *args):
        """
        Set both map and histogram widgets.
        This is a convenience method to update both widgets at once.
        """
        self.set_map(*args)
        self.unmapchange()


    def get_plotly_mapfig(self):
        json = self.geojson
        centroids = []
        for feature in json['features']:
            geom = shape(feature['geometry'])
            centroids.append(geom.centroid.coords[0])
        centroids = np.array(centroids)
        cc = centroids.mean(axis=0)  # (lon, lat)
        # print("Approximate middle:", cc)

        zoomlevel = 11.5
        layout = go.Layout(map_style="carto-positron",
                           map_zoom=zoomlevel,
                           map_center={"lat": cc[1], "lon": cc[0]},  # {"lat": cc[1], "lon": cc[0]},
                           legend_x=0,
                           height=600,
                           width=720,
                           margin=dict(t=10, b=50, l=10, r=150),
                           autosize=True)
        cpmap = go.Choroplethmap(geojson=json,  # json with cell edges
                                locations=[0] * len(json['features']),
                                z=[0] * len(json['features']),
                                colorscale="plasma",
                                showscale=True,
                                marker_line_width=0.5,
                                marker_line_color='gainsboro',
                                marker_opacity=0.8,
                                customdata=[0] * len(json['features']),
                                hovertemplate='<b>%{location}</b><br>' +
                                              '%{customdata}<br>' +  # Only show custom data
                                              '<extra></extra>',
                                name='cpmap')
        fig = go.Figure(cpmap, layout=layout)
        self.set_map(mapfig=fig)
        fig = go.FigureWidget(fig)
        fig.data[0].on_click(self.on_map_click)

        histo = go.Figure(
            [go.Histogram(histnorm='probability density', name=f"iter_{i}", opacity=0.75) for i in
             sorted(self.real_dict.keys())],
            layout=dict(barmode='overlay',
                        height=400,
                        width=500,
                        margin=dict(t=10, b=10, l=10, r=10),
                        yaxis2=dict(overlaying="y", range=[0,1], visible=False))
        )
        histo.add_trace(go.Histogram(
            marker_color='rgba(0,0,0,0)',  # Transparent fill
            marker_line_color='red',  # Outline color
            marker_line_width=2,
            name=f"obs+noise",
            histnorm='probability density',
        ))
        histo.add_trace(go.Scatter(x=[None] * 50, y=np.linspace(0, 1, 50),
                                   mode='lines',
                                   line=dict(color='red', width=3, dash='dash'),
                                   name='obsval',
                                   yaxis='y2',
                                   showlegend=False,
                                   hovertemplate="obsval: %{x}<extra></extra>"))

        histo.add_trace(go.Scatter(x=[None] * 50, y=np.linspace(0,1,50),
                                   mode='lines',
                                   line=dict(color='green', width=3, dash='dash'),
                                   name='mapval',
                                   yaxis='y2',
                                   showlegend=False,
                                   hovertemplate="mapval: %{x}<extra></extra>"))
        histo = go.FigureWidget(histo)
        return fig, histo

    def on_map_click(self, *clickdata):
        trace, p, s = clickdata
        # print(t.locations)
        # get group handler for selected group
        idx = p.point_inds[0]
        print("map index value: ",idx)
        cellid = trace.locations[idx]
        self._cell_sel_id = cellid
        self.highlight_cell()
        with self.maphisto.batch_update():
            self.update_maphisto()


    def highlight_cell(self, mapfig=None):
        """
        Highlight a specific cell in the map.

        Parameters
        ----------
        cellid : int
            The ID of the cell to highlight.
        """
        if mapfig is None:
            mapfig = self.mapwidget
        cellid = self._cell_sel_id
        print("selected cellid :", cellid)
        with mapfig.batch_update():
            trace = mapfig.data[0]
            # Reset all line widths
            # trace.marker.line.width = 0.5
            # trace.marker.line.color = 'gainsboro'
            line_widths = [0.5] * len(trace.locations)
            line_colors = ['gainsboro'] * len(trace.locations)
            if cellid is not None and cellid in trace.locations:
                # Create arrays for line styling
                idx = list(trace.locations).index(cellid)

                # Highlight selected cell
                print("Highlighting cell:", cellid, "at index", idx)
                line_widths[idx] = 2
                line_colors[idx] = 'white'

            else:
                print("No cell selected or cellid not in map data.")
                self._cell_sel_id = None
            trace.marker.line.width = line_widths
            trace.marker.line.color = line_colors

    def _histomod(self, histowgt, df, gp):
        gph = self.obs_dict[gp]
        if self.logselector.value:
            df = np.log10(df)
        for i, dfi in df.groupby('iteration'):
            # print(df)
            histowgt.update_traces(x=dfi.values, selector=dict(name=f"iter_{i}"))
        if gp in self.obsval_dict.keys():
            obsplus = self.obsval_dict[gp]
            obsidx = obsplus.index.intersection(gph.idxmap_r.loc[df.name])
            if len(obsidx) > 0:
                obsplus = obsplus.loc[
                    obsidx].values.flatten()  # todo this will need to change if more than one obs per cell
                if self.logselector.value:
                    obsplus = np.log10(obsplus)
                if len(np.unique(obsplus)) > 1:
                    # only update if there is more than one unique value
                    histowgt.update_traces(x=obsplus,
                                           selector=dict(name=f"obs+noise"))
                    histowgt.update_traces(x=[None]*50, visible=False,
                                           selector=dict(name=f"obsval"))

                else:
                    # no obs+noise for this group
                    print("unique obs+noise value for ", df.name, ":", obsplus)
                    histowgt.update_traces(x=[], selector=dict(name=f"obs+noise"))
                    histowgt.update_traces(x=[obsplus[0]]*50, visible=True,
                                           selector=dict(name=f"obsval"))
            else:
                histowgt.update_traces(x=[], selector=dict(name=f"obs+noise"))
                histowgt.update_traces(x=[None]*50, visible=False,
                                       selector=dict(name=f"obsval"))

        else:
            # no obs+noise for this group
            histowgt.update_traces(x=[], selector=dict(name=f"obs+noise"))
            histowgt.update_traces(x=[None]*50, visible=False,
                                   selector=dict(name=f"obsval"))


    def update_maphisto(self):
        cellid = self._cell_sel_id
        gp = self.mapselector.value
        gph = self.obs_dict[gp]
        if cellid is None or cellid not in gph.ens.index.get_level_values(0):
            self.maphisto.update_traces(x=[])
        else:
            t = self._get_tidx()
            k = self.layselector.value
            # extract the data for the selected cell and tidx
            # -- this has to be a Series
            dff = gph.ens.loc[(cellid, k, [t]), :]
            if len(dff) > 1:
                warnings.warn("Cellid and tidx match more than one output",
                              UserWarning)
            dff = dff.iloc[0]
            self._histomod(self.maphisto, dff, gp)
        self.update_maphisto_line()

    def update_maphisto_line(self):
        cellid = self._cell_sel_id
        rp = self.rpselector.value
        t = self._get_tidx()
        k = self.layselector.value
        i = self.iterselector.value
        if rp == 'r':
            v = self.realselector.value
            data = self.obs_dict[self.mapselector.value].ens
            csel = (i, v)
        else:
            v = self.pslider.value
            data = self.obs_dict[self.mapselector.value].qtiles
            csel = (i, f"P{int(v)}")
        if cellid is None or cellid not in data.index.get_level_values(0):
            dff = None
        else:
            dff = data.loc[(cellid, k, [t]), csel].values[0]
        if self.logselector.value and dff is not None:
            dff = np.log10(dff)
        print("Prob/Real value: ",dff)
        # Update the vertical line in the histogram
        with self.maphisto.batch_update():
            # Remove any existing vertical line
            self.maphisto.update_traces(x=[dff]*50, selector=dict(name=f"mapval"))


    def rpchange(self, change):
        if change.new == 'r':
            self.realselector.disabled = False
            self.pslider.disabled = True
        else:
            self.realselector.disabled = True
            self.pslider.disabled = False
        self.set_map(change)

    def unmapgchange(self, change):
        gsel = self.unmapgroupselector.value
        osel = self.unmapselector.value
        opts = self.obs_dict[gsel].ens.index.to_frame()
        t = self._get_tidx()
        # todo time may or may not be part of this...?
        self.unmapselector.options=opts.loc[opts[self.tidx]==t].index.unique(level=0)
        # if unmapselector.value is changed by the above unmapchange will already have been triggered
        # catch the instance where the value is not changed
        if osel == self.unmapselector.value:
            self.unmapchange()

    def unmapchange(self, change=None):
        gsel = self.unmapgroupselector.value
        v = self.unmapselector.value
        t = self._get_tidx()  # todo time may or may not be part of this...?
        gph = self.obs_dict[gsel]
        seldf = gph.ens.loc[(v, [t]), :]
        if len(seldf) > 1:
            warnings.warn("output and tidx match more than one output",
                          UserWarning)
        seldf = seldf.iloc[0]
        with self.unmaphisto.batch_update():
            self._histomod(self.unmaphisto, seldf, gsel)

    @property
    def default_map_layout(self):
        mapbox = ipyw.VBox([
            ipyw.HTML("<h1>Mappable Obs:</h1>"),
            ipyw.Box([self.mapwidget,
                      ipyw.VBox([self.layselector,
                                 ipyw.HBox([self.cmapselector, self.cmapreverse]),
                                 self.logselector,
                                 self.vminmaxslider,
                                 self.vminmaxbutton,
                                 self.maphisto]),
                      ipyw.VBox([self.wobselector,
                                 self.mapselector]),
                      ipyw.VBox([self.iterselector,
                                 self.rpselector,
                                 ipyw.HBox([self.pslider, self.realselector]),
                                 ])]),
            self.tslider
        ])
        return mapbox

    @property
    def default_unmap_layout(self):
        if self.unmaphisto is None:
            return None
        unmapbox = ipyw.VBox([
            ipyw.HTML("<h1>Unmappable Obs:</h1>"),
            ipyw.Box([self.unmapgroupselector, self.unmapselector]),
            self.unmaphisto
        ])
        return unmapbox

    def set_slider_options(self, index_col='time',
                           options=None, parobs='obs',
                           description="Select:"):
        """
        Set the options for the time-equiv slider.

        Parameters
        ----------
        index_col : str, optional
            The column in the observation dataframe to use for the slider options.
            Default is 'time'.
        options : list, optional
            List of options to set for the slider. If None, unique values in index_col for paramter
            or observation data (according to the value of parobs).
        parobs : str, optional
            The type of data to use for the slider options, either 'obs' or 'par'.
            Default is 'obs'.
        description : str, optional
            The description for the slider widget.
            Default is "Select:".

        """
        if parobs not in ['obs', 'par']:
            raise ValueError("parobs must be either 'obs' or 'par'")
        elif parobs == 'par':
            df = self.pst.parameter_data
        else:
            df = self.pst.observation_data
        if options is None:
            options = sorted(df[index_col].fillna('none').unique().tolist())

        if len(options) < 2:
            self.tslider.disabled = True
        self.tslider.options = [(t,i) for i, t in enumerate(options)]
        self.tslider.value = 0
        self.tslider.description = description
        pass