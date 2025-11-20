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
from ipywidgets import VBox, HBox, Box, Layout

from .utils import _guess_mappable, get_mg_mt, _sort_key, get_geojson

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
            # todo: support cellid already being there
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
                 crs=None,  # coordinate reference system for the modelgrid -- will be converted to WGS84
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
            Optional coordinate reference system for the model grid.
            Only used if geojson is None when it is passed to constructor
            method `mg2geojson` to build json from modelgrid object.
            Defaults to None -- will not attempt to project from model coord
            to lat/lon.
        groupby : str, optional
        tidx: str, optional
        """
        self._callback_off = False
        self._callback_off_count = 0
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

        self.geojson = get_geojson(geojson, mg, crs)

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

        self.map_widget = None
        self.map_histogram = None
        self.unmap_histogram = None
        self.unmap_selector = None
        self.unmap_group_selector = None
        self._build_widgets()
        self._set_widgets()

    @contextmanager
    def callback_off(self):
        self._callback_off_count += 1
        self._callback_off = True
        try:
            yield
        finally:
            self._callback_off_count -= 1
            if self._callback_off_count == 0:
                self._callback_off = False

    def _build_widgets(self):
        # Mapable widgets
        if len(self.gridmapable) > 0 or len(self.pointmapable) > 0:
            self.map_widget, self.map_histogram = self._get_plotly_mapfig() if self.geojson else (None, None)
        # Mappable and observation selection
        self.map_obs_selector = ipyw.RadioButtons(
            options=self.gridmapable,  # list of grid based output groups that can map to json features
            # value=self.gridmapable[0],
            description='Gridded datasets:',
            disabled=False if len(self.gridmapable) > 0 else True,
        )
        self.point_obs_selector = ipyw.RadioButtons(
            options=self.pointmapable,  # list of point based output groups that can map to scatter maps
            # value=self.gridmapable[0],
            description='Scatter datasets:',
            disabled=False if len(self.gridmapable) > 0 else True,
        )
        # Layer selector mappable obs
        self.layer_selector = ipyw.Dropdown(
            options=[], # set later based on selected obs group?
            value=None, # dito
            # description='Layer:',
            disabled=False,
            layout={'width': '100px',
                    'margin': '0px 20px 0px 0px'}
        )
        # Temporal slider for mapping things in time
        self.map_temporal_slider = ipyw.SelectionSlider(
            options=[()], # set later based on selected obs group?
            # value=None, # dito
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            # readout_format='.0f',
        )

        # Colormap selector and reverse checkbox
        self.cmap_selector = ipyw.Dropdown(
            options=pc.named_colorscales(),
            value='Plasma'.lower(),
            description='Colorscale:',
            layout={"justify_content": "flex-start"},
            style={'description_width': 'initial'}
        )
        self.cmap_reverse = ipyw.Checkbox(
            value=False,
            description="Reverse cmap",
            disabled=False,
            layout={"align_self": "flex-start",
                    "justify_content": "flex-start"},
            indent=False
        )
        self.map_log_check = ipyw.Checkbox(value=False,
                                           description="Logscale",
                                           disabled=False,
                                           layout={"align_self": "flex-start",
                                                   "justify_content": "flex-start"},
                                           indent=False)
        # Colormap vmin/vmax slider and reset button
        self.vminmaxslider = ipyw.FloatRangeSlider(
            value=[-1e30, 1e30],
            min=-1e30,
            max=1e30,
            step=100,
            description='cmap range:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            style={"description_width": 'inital'},
        )
        self.vminmaxbutton = ipyw.Button(
            description='Reset range',
            disabled=False,
            button_style='',
            tooltip='Rest vmin/vmax to data range',
            icon="arrows-left-right-to-line"
        )
        # combined widget
        self.vminvmax = VBox([self.vminmaxslider, self.vminmaxbutton])

        # NON mapable widgets
        um = self.unmapable
        ig = None
        if len(um) > 0:
            self.unmap_histogram = self._get_plotly_unmapfig()
            ig = um[0]
        self.unmap_group_selector = ipyw.Dropdown(options=self.unmapable,
                                                  description="Non-mappable groups: ",
                                                  value=ig,
                                                  style={'description_width': 'initial'})
        self.unmap_selector = ipyw.Dropdown(options=[],  # set later
                                            description="Non-mappable obs: ",
                                            value=None,
                                            style={'description_width': 'initial'})

        # Temporal slider for unmapable obs
        self.unmap_temporal_slider = ipyw.SelectionSlider(
            options=[()],
            continuous_update=False,
            orientation='horizontal',
            readout=True,
        )
        # log checkbox for unmapable obs
        self.unmap_log_check = ipyw.Checkbox(value=False,
                                             description="Logscale",
                                             disabled=False,
                                             layout={"align_self": "flex-start",
                                                     "justify_content": "flex-start"},
                                             indent=False)

        # Generic widgets:
        # weighted obs check box
        nnzgps = len(self.obsval_dict.keys())
        self.weighted_obs_checkbox = ipyw.Checkbox(
            value=False,
            description="Weighted only",
            disabled=False if nnzgps > 0 else True,
            indent=False
        )
        # Radio button for selecting realisations of percentiles
        self.reals_or_ptile_radio = ipyw.RadioButtons(
            options=[('Select reals.', 'r'), ("Select P.", 'p')],
            value='p',
            description='Plot type:',
            disabled=False,
        )
        # Slider through probability
        self.prob_slider = ipyw.FloatSlider(
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
        # Iteration and realisation selectors
        self.iter_selector = ipyw.Dropdown(
            options=zip(*[sorted(self.real_dict.keys())] * 2),
            # description="Iteration: ",
            value=sorted(self.real_dict.keys())[0],
            layout={'width': '100px',
                    'margin': '0px 20px 0px 0px'}
        )
        self.real_selector = ipyw.Dropdown(
            options=sorted(self.real_dict[self.iter_selector.value].tolist(), key=_sort_key),
            description="Realisation: ",
            disabled=True, )


    def _get_plotly_mapfig(self):
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
                           # legend_x=0,
                           height=600,
                           # width=720,
                           margin=dict(t=0, b=0, l=0, r=0),
                           autosize=True)
        cpmap = go.Choroplethmap(
            # geojson=json,  # json with cell edges
            # locations=[0] * len(json['features']),
            # z=[0] * len(json['features']),
            # customdata=[0] * len(json['features']),
            geojson=None,  # json with cell edges
            locations=[],
            z=[],
            customdata=[],
            colorscale="plasma",
            showscale=True,
            marker_line_width=0.5,
            marker_line_color='gainsboro',
            marker_opacity=0.8,
            hovertemplate='<b>%{location}</b><br>' +
                          '%{customdata}<br>' +  # Only show custom data
                          '<extra></extra>',
            name='cpmap'
        )
        fig = go.Figure(cpmap, layout=layout)
        # self.set_map(mapfig=fig)
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
        histo.data[0].update(marker_color='rgba(112,112,112,0.75)')
        histo.data[-1].update(marker_color='rgba(20,49,220,0.75)')
        histo.add_trace(go.Histogram(
            marker_color='rgba(0,0,0,0)',  # Transparent fill
            marker_line_color='red',  # Outline color
            marker_line_width=1,
            opacity=0.75,
            name=f"obs+noise",
            histnorm='probability density',
        ))
        histo.add_trace(go.Scatter(x=[None] * 50, y=np.linspace(0, 1, 50),
                                   mode='lines',
                                   line=dict(color='red', width=2, dash='dash'),
                                   opacity=0.75,
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

    def _set_widgets(self):
        """ Setting up widget initial states and callbacks
        """

        self.vminmaxbutton.on_click(self._reset_vminmax)

        self.reals_or_ptile_radio.observe(self.rpchange, names=['value'])

        self.map_obs_selector.observe(self.set_bounds_and_map, names=['value'])

        self.prob_slider.observe(self.set_map, names=['value'])
        self.layer_selector.observe(self.set_map, names=['value'])
        self.iter_selector.observe(self.set_map, names=['value'])
        self.real_selector.observe(self.set_map, names=['value'])
        self.cmap_selector.observe(self.set_map, names=['value'])
        self.cmap_reverse.observe(self.set_map, names=['value'])
        self.map_temporal_slider.observe(self.set_map, names=['value'])

        self.map_log_check.observe(self.set_both, names=['value'])

        self.weighted_obs_checkbox.observe(self.set_mapselector, names=['value'])


        self._reset_vminmax()
        self.vminmaxslider.observe(self.set_vminmax, names=['value'])

        self.unmap_log_check.observe(self.set_unmap, names=['value'])
        self.unmap_temporal_slider.observe(self.set_unmap_options, names=['value'])
        self.unmap_group_selector.observe(self.set_unmap_options, names=['value'])
        self.unmap_selector.observe(self.set_unmap, names=['value'])
        self.set_map()
        if len(self.unmapable) > 0:
            # should trigger set_unmap
            self.set_unmap_options()


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
            mapfig = self.map_widget
        if mapfig is None:
            return
        z = mapfig.data[0].z
        if len(z) == 0:
            return
        self._uservminmax = False
        vmin = z.min()
        vmax = z.max()
        with self.vminmaxslider.hold_trait_notifications():
            self.vminmaxslider.min = vmin
            self.vminmaxslider.max = vmax
            self.vminmaxslider.step = (vmax - vmin) / 1000.0
            self.vminmaxslider.value = [vmin, vmax]

    def _build_obs_handlers(self):
        obs = self.pst.observation_data
        incols = obs.columns.intersection({'kper', 'kstp', 'k', 'i', 'j'})
        obs = obs.astype({c:"Int32" for c in incols})
        if self.tidx == 'time':
            # default is 'time', so infer from kper/kstp if absent
            if 'time' not in obs.columns:
                obs['time'] = np.nan
            # can only do this if we have some reference to build
            # this could be user built ahead of calling this class
            if self.mt is not None and obs.time.isna().any():
                # this will need generalising
                if 'kper' in obs.columns:
                    if 'kstp' not in obs.columns:
                        obs['time'] = obs.time.fillna(
                            obs.apply(
                                lambda x: self.mt.get_elapsed_time(
                                    x.kper if not pd.isna(x.kper) else 0,
                                    None
                                ), axis=1).astype(float)
                        )
                    else:
                        obs['time'] = obs.time.fillna(
                            obs.apply(
                                lambda x: self.mt.get_elapsed_time(
                                    x.kper if not pd.isna(x.kper) else 0,
                                    x.kstp if not pd.isna(x.kstp) else None
                                ), axis=1).astype(float)
                        )
        # At the moment, we want whatever is in tidx to be sortable
        # so all need to be the same dtype
        if obs[self.tidx].apply(pd.api.types.is_number).any():
            try:
                obs[self.tidx] = pd.to_numeric(obs[self.tidx], downcast="integer")
            except Exception:
                obs[self.tidx] = obs[self.tidx].astype(str)
        # fill nans in tidx with 'none' for more
        # reliable grouping and indexing -- need to split out none when sorting later
        obs = obs.fillna({self.tidx: 'none'})
        self.pst.observation_data = obs
        ens = self.pst.ies.obsen.T
        if 'iteration' not in ens.columns.names:
            ens = pd.concat({0: ens}, axis=1, names=['iteration'])
        # handy lookup for realizations for each iteration
        self.real_dict = ens.columns.to_frame(False).groupby('iteration').realization.unique().to_dict()
        try:
            noise = self.pst.ies.noise.T
        except Exception:
            noise = None
        for gp, obdf in obs.groupby(self.groupby):
            gph = VisGroupHandler(obdf, mg=self.mg, ens=ens, tidx=self.tidx)
            if gph.mapable == 'grid':
                self.gridmapable.append(gp)
            elif gph.mapable == 'point':
                self.pointmapable.append(gp)
            else:
                self.unmapable.append(gp)
            if (obdf.weight != 0).any():
                wobs = obdf.loc[obdf.weight != 0]
                # todo: catch and forgive absent noise ensembles
                if noise is not None and len(wobs.index.intersection(noise.index)) > 0:
                    self.obsval_dict[gp] = noise.loc[wobs.index, :]
                else:
                    self.obsval_dict[gp] = pd.DataFrame(index=wobs.index, data=wobs.obsval.values)
            self.obs_dict[gp] = gph
        pass

    def _get_tidx(self, slider):
        t = slider.options[slider.index]
        if not isinstance(t, str) and len(t) > 1:
            t = t[0]
        return t

    def _get_plotly_unmapfig(self):
        unmaphisto = go.Figure(
            [go.Histogram(x=[],
                          histnorm='probability density',
                          name=f"iter_{i}",
                          opacity=0.75)
             for i in sorted(self.real_dict.keys())],
            layout=dict(barmode='overlay',
                        height=400, width=600,
                        margin=dict(t=10, b=10, l=10, r=10),
                        yaxis2=dict(overlaying="y", range=[0,1], visible=False))
        )
        # maker first and last histo grey and blue
        unmaphisto.data[0].update(marker_color='rgba(112,112,112,0.75)')
        unmaphisto.data[-1].update(marker_color='rgba(20,49,220,0.75)')
        # add noise histo
        unmaphisto.add_trace(go.Histogram(
            marker_color='rgba(0,0,0,0)',  # Transparent fill
            marker_line_color='red',  # Outline color
            marker_line_width=0.7,
            opacity=0.75,
            name=f"obs+noise",
            histnorm='probability density',
        ))
        # add vline for obsval
        unmaphisto.add_trace(go.Scatter(
            x=[None]*50, y=np.linspace(0,1,50),
            line=dict(color='red', width=2, dash='dash'),
            opacity=0.75,
            name='obsval',
            yaxis='y2',
            showlegend=False,
            hovertemplate="obsval: %{x}<extra></extra>",
            visible=False
        ))
        # make into widget
        unmaphisto = go.FigureWidget(unmaphisto)
        return unmaphisto


    def set_mapselector(self, change=None):
        with self.callback_off():
            cv = self.map_obs_selector.value
            if self.weighted_obs_checkbox.value:
                gridw = self.obsval_dict.keys() & self.gridmapable
                if len(gridw) > 0:
                    self.map_obs_selector.options = sorted(gridw)
                else:
                    self.weighted_obs_checkbox.value = False
                    self.weighted_obs_checkbox.disabled = True
            else:
                self.map_obs_selector.options = self.gridmapable
            if cv in self.map_obs_selector.options:
                self.map_obs_selector.value = cv
            else:
                self.map_obs_selector.value = self.map_obs_selector.options[0]
        self.set_map(change)

    def set_layselector(self):
        # get current layer selector value
        k = self.layer_selector.value
        # get group handler for selected group
        gp = self.map_obs_selector.value
        gph = self.obs_dict[gp]
        kopt = sorted(gph.metadf.k.unique().tolist())
        with self.callback_off():
            if k is None or k not in kopt:
                self.layer_selector.options = kopt
                self.layer_selector.value = kopt[0]
            else:
                self.layer_selector.options = kopt
                self.layer_selector.value = k
        return self.layer_selector.value

    def set_bounds_and_map(self, change=None):
        self._uservminmax = False
        self.set_map(change=change)

    def set_map(self, change=None, mapfig=None):
        if self._callback_off:
            # if we are in a callback, don't do anything
            return
        # will be used in callback so need to handle change arg
        print("Setting map...")
        if mapfig is None:
            mapfig = self.map_widget
        if mapfig is None:
            return
        # get current group from selector
        gp = self.map_obs_selector.value
        # get group handler for selected group from outputs dict (these contain ensembles etc)
        gph = self.obs_dict[gp]
        # get current selected iteration
        i = self.iter_selector.value
        with self.callback_off():
            self.set_layselector()
            self.set_slider_options(gph.idxmap, self.map_temporal_slider)
            self.real_selector.options = sorted(self.real_dict[i].tolist(), key=_sort_key)
        k = self.layer_selector.value
        t = self._get_tidx(self.map_temporal_slider)
        r = self.real_selector.value
        p = self.prob_slider.value
        log = self.map_log_check.value
        cmap = self.cmap_selector.value
        cr = self.cmap_reverse.value
        if self.weighted_obs_checkbox.value:
            obscells = gph.idxmap.loc[self.obsval_dict[gp].index].cellid.values
        else:
            obscells = slice(None)
        if t in gph.ens.index.unique(self.tidx):
            if self.reals_or_ptile_radio.value == 'r':
                seldf = gph.ens.loc[(obscells, k, t), (i, r)]
            else:
                seldf = gph.qtiles.loc[(obscells, k, t), (i, f"P{int(p)}")]
            z = seldf.values
            locs = seldf.index.get_level_values('cellid')
        else:
            z = []
            locs = []
        if cr:
            cmap += '_r'
        # print(seldf)
        if log:
            z = np.log10(z)

        if self._uservminmax and len(z) > 0:
            zmin, zmax = self.vminmaxslider.value
            zmin = np.max([zmin, z.min()])
            zmax = np.min([zmax, z.max()])
        else:
            zmin, zmax = [None, None]
        print("vminvmax: ", zmin, zmax)

        with mapfig.batch_update():
            mapfig.update_traces(
                geojson=self.geojson,
                z=z,
                zmin=zmin,
                zmax=zmax,
                zauto=True if zmin is None or zmax is None else False,
                locations=locs,
                colorscale=cmap,
                customdata=z,
                selector=dict(name='cpmap')
            )
        if not self._uservminmax:
            with self.callback_off():
                self._reset_vminmax(mapfig=mapfig)
        self.highlight_cell(mapfig)
        if change is not None:
            if change['owner'] == self.real_selector or change['owner'] == self.prob_slider:
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
        if len(self.unmapable) > 0:
            self.set_unmap()

    def on_map_click(self, *clickdata):
        trace, p, s = clickdata
        # print(t.locations)
        # get group handler for selected group
        idx = p.point_inds[0]
        print("map index value: ",idx)
        cellid = trace.locations[idx]
        self._cell_sel_id = cellid
        self.highlight_cell()
        with self.map_histogram.batch_update():
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
            mapfig = self.map_widget
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

    def _histomod(self, histowgt, df, gp, log=False):
        if df is None:
            histowgt.update_traces(x=[])
            return
        gph = self.obs_dict[gp]
        if log:
            df = np.log10(df)
        for i, dfi in df.groupby('iteration'):
            # print(df)
            histowgt.update_traces(x=dfi.values, selector=dict(name=f"iter_{i}"))
        if gp in self.obsval_dict.keys():
            obsplus = self.obsval_dict[gp]
            obsidx = obsplus.index.intersection(gph.idxmap_r.loc[df.name])
            if len(obsidx) > 0:
                obsplus = obsplus.loc[obsidx].values.flatten()  # todo this will need to change if more than one obs per cell
                if log:
                    obsplus = np.log10(obsplus)
                if len(np.unique(obsplus)) > 1:
                    # only update if there is more than one unique value
                    histowgt.update_traces(x=obsplus,
                                           selector=dict(name=f"obs+noise"))
                    histowgt.update_traces(x=[None]*50, visible=False,
                                           selector=dict(name=f"obsval"))

                else:
                    # no obs+noise for this group
                    print("unique obs+noise value for ", df.name, ":", obsplus[0])
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
        gp = self.map_obs_selector.value
        gph = self.obs_dict[gp]
        if cellid is None or cellid not in gph.ens.index.get_level_values(0):
            self.map_histogram.update_traces(x=[])
        else:
            t = self._get_tidx(self.map_temporal_slider)
            k = self.layer_selector.value
            # extract the data for the selected cell and tidx
            # -- this has to be a Series
            dff = gph.ens.loc[(cellid, k, [t]), :]
            if len(dff) > 1:
                warnings.warn("Cellid and tidx match more than one output",
                              UserWarning)
            dff = dff.iloc[0]
            self._histomod(self.map_histogram, dff, gp,
                           log=self.map_log_check.value)
        self.update_maphisto_line()

    def update_maphisto_line(self):
        cellid = self._cell_sel_id
        rp = self.reals_or_ptile_radio.value
        t = self._get_tidx(self.map_temporal_slider)
        k = self.layer_selector.value
        i = self.iter_selector.value
        if rp == 'r':
            v = self.real_selector.value
            data = self.obs_dict[self.map_obs_selector.value].ens
            csel = (i, v)
        else:
            v = self.prob_slider.value
            data = self.obs_dict[self.map_obs_selector.value].qtiles
            csel = (i, f"P{int(v)}")
        if cellid is None or cellid not in data.index.get_level_values(0):
            dff = None
        else:
            dff = data.loc[(cellid, k, [t]), csel].values[0]
        if self.map_log_check.value and dff is not None:
            dff = np.log10(dff)
        print("Prob/Real value: ",dff)
        # Update the vertical line in the histogram
        with self.map_histogram.batch_update():
            # Remove any existing vertical line
            self.map_histogram.update_traces(x=[dff] * 50, selector=dict(name=f"mapval"))


    def rpchange(self, change):
        if change.new == 'r':
            self.real_selector.disabled = False
            self.prob_slider.disabled = True
        else:
            self.real_selector.disabled = True
            self.prob_slider.disabled = False
        self.set_map(change)


    def set_unmap_options(self, change=None):
        if self._callback_off:
            # if we are in a callback, don't do anything
            return
        # if group selector changes, need to update obs selector options
        # dependent on the selected time (this can get circular!)
        print("Setting unmap options...")
        gsel = self.unmap_group_selector.value
        osel = self.unmap_selector.value
        gph = self.obs_dict[gsel]
        # if unmap group has changed need to revaluate temporal slider options
        with self.callback_off():
            self.set_slider_options(gph.idxmap, self.unmap_temporal_slider)
        opts = self.obs_dict[gsel].ens.index.to_frame()
        t = self._get_tidx(self.unmap_temporal_slider)
        # todo time may or may not be part of this...?
        self.unmap_selector.options=opts.loc[opts[self.tidx] == t].index.unique(level=0)
        if osel in self.unmap_selector.options:
            self.unmap_selector.value = osel
        else:
            self.unmap_selector.value = self.unmap_selector.options[0]
        # if unmapselector.value is changed by the above set_unmap will already have been triggered
        # catch the instance where the value is not changed
        if osel == self.unmap_selector.value:
            self.set_unmap()

    def set_unmap(self, change=None):
        if self._callback_off:
            # if we are in a callback, don't do anything
            return
        # if unmap observation changed need to update histogram
        print("Setting unmap...")
        gsel = self.unmap_group_selector.value
        gph = self.obs_dict[gsel]
        t = self._get_tidx(self.unmap_temporal_slider)  # todo time may or may not be part of this...?
        v = self.unmap_selector.value
        try:
            seldf = gph.ens.loc[(v, [t]), :]
            if len(seldf) > 1:
                warnings.warn("output and tidx match more than one output",
                              UserWarning)
            seldf = seldf.iloc[0]
        except KeyError:
            seldf = pd.DataFrame()
        with self.unmap_histogram.batch_update():
            self._histomod(self.unmap_histogram, seldf, gsel,
                           log=self.unmap_log_check.value)

    @property
    def default_map_layout(self):
        # layer select box (label on top)
        laysel_box = VBox(
            [ipyw.Label("Layer:",
                        layout={'align_self':'flex-start'}),
             self.layer_selector],
            layout=Layout(justify_content='space-around')
        )
        csel_box = VBox(
            [HBox([self.cmap_selector, self.cmap_reverse, self.map_log_check],
                  layout=Layout(width='100%',
                                display='flex',
                                justify_content='space-around',
                                align_content='flex-start',
                                align_items='flex-start')),
             HBox([self.vminmaxslider, self.vminmaxbutton])],
             layout=Layout(width='100%',
                           display='flex',
                           justify_content='space-around',
                           align_content='flex-start',
                           align_items='flex-start',
                           # border='2px solid black'
                           )
             )
        sel0_box = Box([laysel_box, csel_box],
                       layout=Layout(width='100%',
                                     display='flex',
                                     justify_content='center',
                                     align_content='flex-start',
                                     align_items='flex-start',
                                     justify_items='center',
                                     border='1px solid grey'))
        iter_box = Box([ipyw.Label("Iteration:",
                        layout={'align_self':'flex-start'}),
                        self.iter_selector],
                       layout=Layout(justify_content='flex-start'))
        sel2_box = Box([VBox([self.weighted_obs_checkbox,
                              self.map_obs_selector]),
                        VBox([iter_box,
                              self.reals_or_ptile_radio,
                              HBox([self.prob_slider, self.real_selector])])],
                       layout=Layout(
                width='100%',
                # height='260px',
                # background='#52B5E8',
                display='flex',
                justify_content='space-around',
                align_items='flex-start',
                border='1px dashed grey'
            )
                       )
        map_box = Box(
            [self.map_widget],
            layout=Layout(
                flex='1 1 auto',
                width='100%',
                # min_height='600px',
                # min_width='850px',
                # background='#52B5E8',
                display='flex',
                justify_content='center',
                align_items='center',
                # border='1px solid black',
                margin="0px 10px 0px 0px"
            )
        )

        slider_box = Box(
            [self.map_temporal_slider],
            layout=Layout(
                width='100%',
                height='60px',
                # background='#52B5E8',
                display='flex',
                justify_content='center',
                align_items='center',
                border='3px solid white'
            )
        )

        histo_box = Box(
            [self.map_histogram],
            layout=Layout(
                width='100%',
                min_height='400px',
                # background='#52B5E8',
                display='flex',
                justify_content='center',
                align_items='center',
                border='3px solid white'
            )
        )
        # Create left column (Selector0, MapBox, SliderBox)
        left_column = VBox(
            [sel0_box, map_box, slider_box],
            layout=Layout(
                flex='1 1 60%',
                min_width='300px',
                display='flex',
                flex_flow='column'
            )
        )

        right_column = VBox(
            [sel2_box, histo_box],
            layout=Layout(
                flex='1 1 40%',
                min_width='300px',
                display='flex',
                flex_flow='column'
            )
        )

        content_row = HBox(
            [left_column, right_column],
            layout=Layout(
                width='100%',
                display='flex',
                flex_flow='row wrap',
                align_items='flex-start',
            )
        )

        main_container = VBox(
            [ipyw.HTML("<h1>Mappable Obs:</h1>"), content_row],
            layout= Layout(
                width='100%',
                height='100%',
                display='flex',
                flex_flow='column'
            )
        )
        return main_container

    @property
    def default_unmap_layout(self):
        if self.unmap_histogram is None:
            return None
        unmapbox = ipyw.VBox([
            ipyw.HTML("<h1>Unmappable Obs:</h1>"),
            ipyw.Box([self.unmap_group_selector, self.unmap_selector]),
            ipyw.Box([self.unmap_histogram, ipyw.VBox([self.unmap_temporal_slider,
                                                       self.unmap_log_check])])
        ])
        return unmapbox

    def set_slider_options(self, idxs, slider=None, description="Time:"):
        """
        Set slider options based on unique values in observation or parameter data
        Parameters
        ----------
        slider : ipyw.SelectionSlider, optional
            Slider widget to set options for. If None, uses self.map_temporal_slider
        description : str, optional
            Description for the slider widget. Default is "Time:"

        Returns
        -------

        """
        if slider is None:
            slider = self.map_temporal_slider
        t = self._get_tidx(slider)
        options = idxs[self.tidx].fillna('none').unique().tolist()
        isnone = True
        try:
            options.remove('none')
        except ValueError:
            isnone = False
        options = sorted(options)
        if isnone:
            options = ['none'] + options

        if len(options) < 2:
            slider.disabled = True
        i = options.index(t) if t in options else 0
        options = [(t, i) for i, t in enumerate(options)]
        slider.options = options
        slider.value = i
        slider.description = description
        pass