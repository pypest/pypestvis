"""Core functionality of pypestvis"""

__all__ = ["VisHandler"]

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

from .utils import (_guess_mappable, get_mg_mt,
                    _sort_key, get_geojson, get_cellid_fromij,
                    _nat_sort)


class VisHandler(object):
    """
    Handler for visualizations in the web application. Currently flopy mf6
    modelgrid dependent.
    """
    def __init__(self,
                 pst,
                 geojson=None,  # needed for mapping, could add additional geojson options as extra kwargs?
                 wd=None,  # working directory for the model, needed to get mg from grb, also a save location
                 mg=None,  # needed for referencing kij to json, also can be used to build json if geojson is absent
                 mt=None,  # model time, needed for obs data
                 crs=None,  # coordinate reference system for the modelgrid -- will be converted to WGS84
                 groupby='obgnme',  # groupby for the obs data, default is obgnme
                 tidx='time',
                 locidx=None,  # location index for identifying unique obs locations
                 write_json=False,):
        """

        Parameters
        ----------
        pst : pyemu.Pst or str or Path
            Pest control file for project
        geojson : str or json, optional
            Geojson file dict for model grid features for plotting
        wd : str or Path, optional
            Project working directory for the model. Needed if either mg or mt
            are None.
        mg : flopy.ModelGrid, optional
            Flopy (for now) model grid definition for the model. Needed if
            geojson is None.
        mt : flopy.ModelTime, optional
            Flopy (for now) model time definition for the model. Needed for
            time referencing of obs data.
        crs : str, optional
            Coordinate reference system for the model grid.
            Only used if geojson is None when it is passed to constructor
            method `mg2geojson` to build json from modelgrid object.
            Defaults to None -- will not attempt to project from model coord
            to lat/lon.
        groupby : str, optional
            Column name in pst.observation_data to group observations by.
            Default is 'obgnme'.
        tidx : str, optional
            Column name in pst.observation_data to use as temporal index for
            time slider. Default is 'time'.
        write_json : bool, optional
            Write geojson to file if geojson is None and
            needs to be built from modelgrid. Default is False.
            If True will write to assets/{pstname}_modelgrid.json as default
            (supposedly for faster browser cacheing).
        """

        self.__pst = pst
        self.__geojson = geojson
        self.__wd = wd
        self.__mg = mg
        self.__mt = mt
        self.__crs = crs
        self.__groupby = groupby
        self.__tidx = tidx
        self.__write_json = write_json

        self.crs = crs

        self._callback_off = False
        self._callback_off_count = 0

        # Bring in control file -- lean on pyemu a lot
        if isinstance(pst, (str, Path)):
            pst = pyemu.Pst(str(pst))
        self.pst = pst
        self.name = Path(pst.filename).stem

        # get spatial and temporal reference -- currently leaning on
        # flopy and mf6 modelgrid/time
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
        # carry temporal slider index column name
        self.tidx = tidx
        self.locidx = locidx

        # need a geojson for mapping
        if geojson is None:
            # set to default value
            geojson = Path("assets", f"{self.name}_modelgrid.json")
        self.geojson = get_geojson(geojson=geojson,
                                   mg=mg,
                                   crs=crs,
                                   wd=wd,
                                   write=write_json)

        # lists for storing tags of mappable status of data groups
        self.gridmapable = []
        self.pointmapable = []
        self.unmapable = []
        # self.weighted = []
        self.lines = None

        # carry grouping column
        self.groupby = groupby

        # initial widget placeholders
        self.map_widget = None
        self.map_histogram = None
        self.unmap_histogram = None

        # setup empty containers
        self.obs_gphandlers = {}
        # self.obsval_dict = {}
        self.par_dict = {}  # not yet implementing pars
        self.real_dict = {}  # set in _build_obs_handlers()
        self._build_obs_handlers()

        # initial setting for callback status
        self._tmp_map_gph = None
        self._tmp_map_kidxmap = None # current index map for mapped data at all tidxs
        self._tmp_map_idxmap = None
        # self._tmp_map_df = None # current dataframe for mapped data at selected tidx
        self._tmp_map_ens = None # current ensemble for mapped data at selected tidx
        self._sel_cellid = None
        self._sel_name = None
        self._sel_ensdf = None
        self._uservminmax = False # for storing if user has set vmin/vmax

        # build widgets and setup callbacks
        self._build_widgets()
        self._set_widget_callbacks()

        # trigger map build
        if len(self.gridmapable) > 0 or len(self.pointmapable) > 0:
            # should trigger cascade through to set_map
            self.set_mapsel_options()
            # finalise map widget as FigureWidget for interactivity
            self.map_widget = go.FigureWidget(self.map_widget)
            self.map_widget.data[0].on_click(self.on_map_click)
            self.map_widget.data[1].on_click(self.on_map_click)
        if len(self.unmapable) > 0:
            # should trigger set_unmap
            self.set_unmap_group()
            # make into widget
            self.unmap_histogram = go.FigureWidget(self.unmap_histogram)

    def __str__(self):
        return (f"VisHandler for Pst: '{str(self.name)}' with {len(self.obs_gphandlers)} obs groups:\n"
                '\n'.join(self.obs_gphandlers.keys()))


    def __repr__(self):
        return ("VisHandler(\n"
                fr"    pst={str(self.__pst)},"'\n'
                fr"    geojson={str(self.__geojson)},"'\n'
                fr"    wd='{str(self.__wd)}',"'\n'
                fr"    mg={str(self.__mg)},"'\n' 
                fr"    mt={str(self.__mt)}," + '\n'
                fr"    crs='{str(self.__crs)}',"'\n'
                fr"    groupby='{str(self.groupby)}',"'\n'
                fr"    tidx='{str(self.tidx)}',"'\n'
                fr"    locidx='{str(self.locidx)}',"'\n'                             
                fr"    write_json={str(self.__write_json)}"'\n'
                ')')


    class VisGroupHandler(object):
        """
        Handler Class for groups in the web application.
        """

        def __init__(self, parent, df, gpname, ens=None, obsplus=None):
            """
            Parameters
            ----------
            parent : VisHandler
                Parent VisHandler object
            df : pd.DataFrame
            ens : pd.DataFrame, optional
                Ensemble indexed by obs/par names with columns as multiindex
                of (iterations, realization).
            obsplus : pd.DataFrame, optional
                Ensemble of observation noise values
                indexed by obs names with columns as multiindex of
                (iterations, realization).
            """
            self.__parent = parent
            self.__df = df
            self.__ens = ens
            self.__obsplus = obsplus

            mg = parent.mg
            tidx = parent.tidx
            locidx = parent.locidx
            self.name = gpname
            layer_col = 'k'
            # check status of i,j columns
            if df.i.notna().all() & df.j.notna().all():
                # make sure ij etc are int32 for indexing
                # dont need to be nullable int and better for downstream
                # numpy methods
                df = df.astype({'i': int, 'j': int})
            # work out if grid, point, or unmap outputs
            self.mapable = _guess_mappable(df)

            if self.mapable == 'grid':  # can place on a grid
                # update parent class with name of group
                parent.gridmapable.append(gpname)
                # need to get cellid from so we can ref to json
                # todo: support cellid already being there
                df['cellid'] = get_cellid_fromij(tuple(df[['i', 'j']].values.T), mg.shape[1:])
                idxcols = ['cellid', 'k', tidx]
            elif self.mapable == 'point':
                # may not be fully implemented but would need x,y locations
                # to map to scatter
                # (note may need a projection step here or later)
                parent.pointmapable.append(gpname)
                if 'x' not in df.columns:
                    df['x'] = np.nan
                if 'y' not in df.columns:
                    df['y'] = np.nan
                df = df.fillna(pd.DataFrame(
                    {'x': mg.xcellcenters[df.i.values, df.j.values],
                     'y': mg.ycellcenters[df.i.values, df.j.values]},
                    index=df.index
                ))
                if parent.crs is not None:
                    import geopandas as gpd
                    df[['x', 'y']] = gpd.GeoSeries.from_xy(
                        *df[['x', 'y']].values.T,
                        crs=parent.crs,
                        index=df.index
                    ).to_crs('epsg:4326').get_coordinates()
                if locidx is None:
                    _locidx = 'site'
                else:
                    _locidx = locidx
                if _locidx not in df.columns or df[_locidx].isna().all():
                    df[_locidx] = pd.factorize(pd._libs.lib.fast_zip(
                        [df.x.values,df.y.values]))[0].astype(str)
                idxcols = [_locidx, 'x', 'y', 'k', tidx]
            else:
                # update parent class with name of group
                parent.unmapable.append(gpname)
                # assuming usecol is unique identifier for unmapable obs
                if locidx is None:
                    usecol = 'usecol'
                else:
                    usecol = locidx
                idxcols = [usecol, tidx]

            # make sure that there is something in the time index columns
            # incols = df.columns.intersection({'kper', 'kstp', 'k', 'i', 'j'})
            # obs = obs.astype({c: "Int32" for c in incols})
            if tidx == 'time': # tidx passed to parent class
                # default is 'time', so infer from kper/kstp if absent
                if 'time' not in df.columns:
                    df['time'] = np.nan
                # df['time'] = df.time.astype(float)
                # can only do this if we have some reference to build
                # this could be user built ahead of calling this class
                if parent.mt is not None and df.time.isna().any():
                    # if we have time translation info
                    # and if and null in that column
                    # this will need generalising
                    if 'kper' in df.columns:
                        # nullable pandas int
                        kperkstp = df.kper.astype("Int32").fillna(0).to_frame() # fill na with 0 for now
                        if 'kstp' in df.columns:
                            kperkstp['kstp'] = df.kstp.astype("Int32")
                        else:
                            kperkstp['kstp'] = np.nan
                        kperkstp.loc[kperkstp.kstp.isna(), 'kstp'] = (parent.mt.nstp[kperkstp.kper] - 1)[kperkstp.kstp.isna()]
                        kperkstp = kperkstp.astype("Int32")
                        # fill nans
                        df.loc[df.time.isna(), 'time'] = [
                            parent.mt.get_elapsed_time(per, stp).astype(df.time.dtype.type)
                            for per,stp in kperkstp.loc[df.time.isna()].values
                        ]
            # At the moment, we want whatever is in tidx to be sortable
            # so all need to be the same dtype
            try:
                df[tidx] = pd.to_numeric(df[tidx], downcast="integer",
                                         errors="raise")
            except ValueError:
                # fill nans in tidx with 'none' for more
                # reliable grouping and indexing -- need to split out none when sorting later
                df[tidx] = df[tidx].astype(str)
                df = df.fillna({tidx: 'none'})

            if ens is None:
                self.ens = None
                self.qtiles = None
            else:
                # slice out group from ensemble
                gpens = ens.loc[df.index, :].copy()
                # set ensemble index to multiindex
                # actually, let's leave this as obs/parnames
                # gpens.index = pd.MultiIndex.from_frame(df[idxcols])
                self.ens = gpens
                self.qtiles = self.ens.T.groupby(level='iteration').quantile(
                    np.linspace(start=0, stop=1, num=21)
                ).T
                # rename percentiles
                self.qtiles.columns = self.qtiles.columns = self.qtiles.columns.set_levels(
                    self.qtiles.columns.levels[1].map(lambda x: int(100 * x)),
                    level=1
                )
            weighted = df.weight != 0
            if (weighted).any():
                wobs = df.loc[weighted]
                # todo: catch and forgive absent noise ensembles
                if obsplus is not None and len(wobs.index.intersection(obsplus.index)) > 0:
                    self.obsplus = obsplus.loc[wobs.index, :]
                else:
                    self.obsplus = pd.DataFrame(index=wobs.index,
                                                data=wobs.obsval.values)
            else:
                self.obsplus = None

            # set the index columns for fast lookups in interaction later
            # this is going to form the basis for the onclick actions
            # will slice by the second last level (layer) on layer selection
            # then slice by the last level (time) on time slider change
            self.group_info = df.reset_index(names=['ensmap']).set_index(idxcols)

        def __str__(self):
            return (f"VisGroupHandler for '{self.name}' with {len(self.group_info)} obs\n"
                    f"Mapable status: {self.mapable}\n"
                    f"Ensemble members: {self.ens.shape[1] if self.ens is not None else 0}\n"
                    f"Weighted obs: {self.obsplus.shape[0] if self.obsplus is not None else 0}")

        def __repr__(self):
            return self.__str__() + "\n" + str(self.group_info)

    # Group handler construction
    def _build_obs_handlers(self):
        obs = self.pst.observation_data
        ens = self.pst.ies.obsen.T
        if 'iteration' not in ens.columns.names:
            ens = pd.concat({0: ens}, axis=1, names=['iteration'])
        # handy lookup for realizations for each iteration
        self.real_dict = ens.columns.to_frame(False).groupby('iteration').realization.unique().to_dict()
        try:
            noise = self.pst.ies.noise.T
        except Exception:
            noise = None
        self.nonzero_groups = []
        for gp, obdf in obs.groupby(self.groupby):
            gph = self.VisGroupHandler(self, obdf, gp, ens=ens, obsplus=noise)
            self.obs_gphandlers[gp] = gph
            if gph.obsplus is not None and len(gph.obsplus) > 0:
                self.nonzero_groups.append(gp)

    # Widget Construction
    def _build_widgets(self):
        # Mapable widgets
        mappable = list(self.gridmapable) + list(self.pointmapable)
        if len(mappable) > 0:
            if self.geojson:
                self._build_mappable_plotly()

        # Mappable and observation selection
        self.map_obs_selector = ipyw.RadioButtons(
            options=mappable,  # list of grid based output groups that can map to json features
            # value=self.gridmapable[0],
            description='Mappable datasets:',
            disabled=False if len(mappable) > 0 else True,
        )
        # self.point_obs_selector = ipyw.RadioButtons(
        #     options=self.pointmapable,  # list of point based output groups that can map to scatter maps
        #     # value=self.gridmapable[0],
        #     description='Scatter datasets:',
        #     disabled=False if len(self.pointmapable) > 0 else True,
        # )
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
            self._set_plotly_unmapfig()
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
        nnzgps = len(self.nonzero_groups)
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

    def _set_widget_callbacks(self):
        """ Setting up widget initial states and callbacks
        """
        # Ensemble selection callbacks
        self.iter_selector.observe(self.set_ensemble, names=['value'])
        self.reals_or_ptile_radio.observe(self.set_ensemble, names=['value'])

        # Map widget callbacks
        self.weighted_obs_checkbox.observe(self.set_mapsel_options, names=['value'])
        self.map_obs_selector.observe(self.select_map_obs_gp, names=['value'])
        self.layer_selector.observe(self.select_map_layer, names=['value'])
        self.map_temporal_slider.observe(self.set_map, names=['value'])
        self.map_log_check.observe(self.set_map, names=['value'])
        self.prob_slider.observe(self.set_map, names=['value'])
        self.real_selector.observe(self.set_map, names=['value'])

        # Map cbar mods -- needs more attention
        self.vminmaxbutton.on_click(self._reset_vminmax)
        self.cmap_selector.observe(self.set_map, names=['value'])
        self.cmap_reverse.observe(self.set_map, names=['value'])
        self._reset_vminmax()
        self.vminmaxslider.observe(self.set_vminmax, names=['value'])

        # Unmap widget callbacks
        self.unmap_group_selector.observe(self.set_unmap_group, names=['value'])
        self.unmap_selector.observe(self.set_unmap_level, names=['value'])
        self.unmap_temporal_slider.observe(self.set_unmap, names=['value'])
        self.unmap_log_check.observe(self.set_unmap, names=['value'])

    def _build_mappable_plotly(self):
        from itertools import cycle
        from plotly.colors import DEFAULT_PLOTLY_COLORS
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
            z=np.array([]),
            customdata=[],
            colorscale="plasma",
            showscale=True,
            marker_line_width=0.5,
            marker_line_color='gainsboro',
            marker_opacity=0.8,
            hovertemplate='<b>%{meta}</b><br>' +
                          '%{customdata}<br>' +  # Only show custom data
                          '<extra></extra>',
            name='cpmap'
        )

        scatmap = go.Scattermap(
            lat=[],
            lon=[],
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.7,
                showscale=True,
            ),
            # selected=go.scattermap.Selected(
            #     marker=dict(opacity=1, size=12)
            # ),
            customdata=[],
            meta=[],
            hovertemplate='<b>%{meta}</b><br>' +
                          '%{customdata}<br>' +  # Only show custom data
                          '<extra></extra>',
            name='scatmap'
        )

        # TODO: add scatter widget for point mappables!!
        # leave as Figure object until after first
        # update_traces call for voila compat.
        fig = go.Figure([cpmap,scatmap], layout=layout)

        histo = go.Figure(
            [go.Histogram(histnorm='probability density', name=f"iter_{i}", opacity=0.75) for i in
             sorted(self.real_dict.keys())],
            layout=dict(barmode='overlay',
                        height=400,
                        width=500,
                        margin=dict(t=10, b=10, l=10, r=10),
                        yaxis2=dict(overlaying="y", range=[0,1], visible=False))
        )
        histo.data[-1].update(marker_color='rgba(20,49,220,0.75)')
        histo.data[0].update(marker_color='rgba(112,112,112,0.75)')
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

        tsplot = go.Figure(layout=dict(margin_t=30,
                                       margin_b=10,
                                       margin_l=10,
                                       margin_r=10,
                                       width=600,
                                       height=300,
                                       title='T-series',
                                       showlegend=True,
                                       xaxis_autorange=True))
        tsplot.add_vline(x=0,
                         line_color='black',
                         line_dash='dash',
                         annotation_text="selected time",
                         annotation_name="selected time",
                         annotation_visible=False,
                         name="tsel",
                         visible=True,
                         showlegend=True)
        tsplot = go.FigureWidget(tsplot)



        self.map_widget = fig
        self.map_histogram = histo
        self.map_ts = tsplot

    # Callbacks
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

    # weighted_obs_checkbox callback at top of map obs selection chain
    def set_mapsel_options(self, change=None):
        """
        Call back function triggered by weighted_obs_checkbox changes.
        Updates of map_obs_selector options and triggers downstream updates of:
        select_map_obs_gp() -> callback for map_obs_selector changes
          _tmp_map_gph -> group handler for current group selection
          _set_laysel_options(self)
            layer_selector.options -> current layer options for mapping
            select_map_layer()
              _tmp_map_kidxmap -> index/name map for current mapping layer (including time)
              _set_slider_options()
              set_map()
                if _tmp_map_ens is None:
                    set_ensemble()
                      _tmp_map_ens -> current iteration ensemble for mapping
                      real_selector.options -> current realisation options iteration
                _tmp_map_idxmap() -> index/name map for current mapping layer and time
                _set_sel_name()
                  _sel_name -> current selected (obs) name (may be None)
                highlight_cell() [maybe set _sel_cellid = None]
                update_maphisto_line()
                  or
                update_maphisto()
                  _histomod()
                  update_maphisto_line()
        Parameters
        ----------
        change

        Returns
        -------

        """
        if self._callback_off:
            return
        # called when weighted obs checkbox change
        if self.weighted_obs_checkbox.value:
            print("Toggle on weighted obs...")
        else:
            print("Toggle off weighted obs...")
        with self.callback_off():  # don't want to trigger all the callbacks?
            # current map_obs_selector value
            cv = self.map_obs_selector.value
            if self.weighted_obs_checkbox.value:
                # get weighted groups that are gridmapable
                gridw = set(self.nonzero_groups) & set(self.gridmapable + self.pointmapable)
                if len(gridw) > 0:
                    # update options to only weighted
                    opts = sorted(gridw)
                    self.weighted_obs_checkbox.disabled = False
                else:
                    # if none then reset and disable checkbox
                    self.weighted_obs_checkbox.value = False
                    self.weighted_obs_checkbox.disabled = True
                    opts = self.gridmapable + self.pointmapable
            else:
                # reset to all gridmapable
                opts = self.gridmapable + self.pointmapable
            # fix value to current if possible
            self.map_obs_selector.options = opts
            if cv in opts:
                # make sure value stays the same if possible
                self.map_obs_selector.value = cv
            else:
                self.map_obs_selector.value = self.map_obs_selector.options[0]
        # propagate now
        self.select_map_obs_gp(change=change)

    # map_obs_selector callback
    def select_map_obs_gp(self, change=None):
        """
        Call back function triggered by map_obs_selector changes.
        Updates:
        _tmp_map_gph -> group handler for current group selection
        Triggers downstream:
        _set_laysel_options(self)
          layer_selector.options -> current layer options for mapping
          select_map_layer()
            _tmp_map_kidxmap -> index/name map for current mapping layer (including time)
            _set_slider_options()
            set_map()
              if _tmp_map_ens is None:
                set_ensemble()
                  _tmp_map_ens -> current iteration ensemble for mapping
                  real_selector.options -> current realisation options iteration
              _tmp_map_idxmap() -> index/name map for current mapping layer and time
              _set_sel_name()
              _sel_name -> current selected (obs) name (may be None)
              highlight_cell() [maybe set _sel_cellid = None]
              update_maphisto_line()
                or
              update_maphisto()
                _histomod()
                update_maphisto_line()
        Parameters
        ----------
        change

        Returns
        -------

        """
        if self._callback_off:
            # if we are in a callback, don't do anything
            return
        # called when map_obs_selector value changes
        # update group handler loaded for mapping
        gp = self.map_obs_selector.value
        print(f"Extracting outputs for {gp}")
        self._tmp_map_gph = self.obs_gphandlers[gp]
        self.set_ensemble(propagate=False)
        # TODO: check vminvmax defaults
        self._uservminmax = False
        # propagate through to layer selection options
        self._set_laysel_options(change=change)
        # this will propagate to layer seleciton
        # and finally to tslider definition...

        # self.set_map(change=change)

    # map_obs_selector internal callback
    def _set_laysel_options(self, change=None):
        """
        Internal callback to set layer selector options based on
        current group handler and weighted obs checkbox status.
        Updates:
        layer_selector.options -> current layer options for mapping
        Triggers downstream:
        select_map_layer()
          _tmp_map_kidxmap -> index/name map for current mapping layer (including time)
          _set_slider_options()
          set_map()
            if _tmp_map_ens is None:
              set_ensemble()
                _tmp_map_ens -> current iteration ensemble for mapping
                real_selector.options -> current realisation options iteration
            _tmp_map_idxmap() -> index/name map for current mapping layer and time
            _set_sel_name()
              _sel_name -> current selected (obs) name (may be None)
            highlight_cell() [maybe set _sel_cellid = None]
            update_maphisto_line()
              or
            update_maphisto()
              _histomod()
              update_maphisto_line()

        Returns
        -------

        """
        # called downstream when map obs group changes
        # get current layer selector value
        o_k = self.layer_selector.value
        # get group handler for selected group
        gph = self._tmp_map_gph # is is already set on obs gp change
        print(f"Setting layer options for {gph.name}...")
        lookup = gph.group_info
        # expecting layer index level -2
        if self.weighted_obs_checkbox.value:
            # filter to weighted only
            lookup = lookup[lookup.weight != 0]
        kopt = _nat_sort(lookup.index.unique(-2))
        if len(kopt) > 1 and self._tmp_map_gph.mapable == 'point':
            kopt = ['all'] + kopt
        # update options and values
        # WILL TRIGGER LAYER SELECTOR CALLBACK if ok changes
        if o_k is None or o_k not in kopt:
            self.layer_selector.options = kopt
            self.layer_selector.value = kopt[0]
        else:
            with self.callback_off():
                self.layer_selector.options = kopt
                self.layer_selector.value = o_k
            self.select_map_layer(change=change)

    # layer_selector callback
    def select_map_layer(self, change=None):
        """
        Call back function triggered by layer_selector changes.
        Updates:
        _tmp_map_kidxmap -> index/name map for current mapping layer (including time)
        Triggers downstream:
        _set_slider_options()
        set_map()
          if _tmp_map_ens is None:
            set_ensemble()
              _tmp_map_ens -> current iteration ensemble for mapping
              real_selector.options -> current realisation options iteration
          _tmp_map_idxmap() -> index/name map for current mapping layer and time
          _set_sel_name()
            _sel_name -> current selected (obs) name (may be None)
          highlight_cell() [maybe set _sel_cellid = None]
          update_maphisto_line()
            or
          update_maphisto()
          _histomod()
          update_maphisto_line()

        Parameters
        ----------
        change

        Returns
        -------

        """
        # called when layer selector changes
        k = self.layer_selector.value
        if k == 'all' and self._tmp_map_gph.mapable == 'point':
            k = slice(None)
        gp = self._tmp_map_gph.name  # is is already set on obs gp change
        print(f"Extracting layer data for {gp}@k:{k}...")
        mapdf = self._tmp_map_gph.group_info
        nlev = mapdf.index.nlevels
        kslice = (slice(None),) * (nlev - 2) + (k,) + (slice(None),)
        kdf = mapdf.loc[kslice, :].droplevel(-2)
        if self.weighted_obs_checkbox.value:
            # filter to weighted only
            kdf = kdf[kdf.weight != 0]
        # DEFINE INDEX MAP FOR SELECTED LAYER
        # -- this is ths key attribute for slicing at runtime
        self._tmp_map_kidxmap = kdf.ensmap
        # propagate through to temporal slider
        with self.callback_off():
            self._set_slider_options(self.map_temporal_slider,
                                     self._tmp_map_kidxmap)
        # this now?
        self.set_map(change=change)

    # layer_selector internal callback
    def _set_slider_options(self, slider=None, idxmap=None,
                            description="Time:"):
        if slider is None:
            slider = self.map_temporal_slider
        if idxmap is None:
            idxmap = self._tmp_map_kidxmap
        print("Setting slider options...")
        t = self._get_tidx(slider)
        options = idxmap.index.unique(self.tidx).tolist()
        isnone = True
        try:
            options.remove('none')
        except ValueError:
            isnone = False
        options = _nat_sort(options)
        if isnone:
            options = ['none'] + options

        if len(options) < 2:
            slider.disabled = True
        else:
            slider.disabled = False
        val = options.index(t) if t in options else 0
        options = [(t, i) for i, t in enumerate(options)]
        slider.options = options
        slider.value = val
        slider.description = description

    def set_ensemble(self, change=None, propagate=True):
        i = self.iter_selector.value
        gp = self._tmp_map_gph.name
        # PRELOADING ENSEMBLE AT ITERATION
        if self.reals_or_ptile_radio.value == 'r':
            print(f"Loading {gp} realisation ensemble for iter {i}...")
            ens = self._tmp_map_gph.ens.xs(i, level=0, axis=1)
            self.real_selector.disabled = False
            self.prob_slider.disabled = True
        else:
            print(f"Loading {gp} quantiles for iter {i}...")
            ens = self._tmp_map_gph.qtiles.xs(i, level=0, axis=1)
            self.real_selector.disabled = True
            self.prob_slider.disabled = False
        # realisation or quantile ensemble at selected iteration
        self._tmp_map_ens = ens

        with self.callback_off():
            # self._set_slider_options(self.map_temporal_slider)
            self.real_selector.options = sorted(self.real_dict[i].tolist(), key=_sort_key)

        if propagate:
            self.set_map(change=change)

    def set_map(self, change=None, mapfig=None):
        """
        Call back function to set map data onto map widget based on current selections.
        Calls downstream:
        if _tmp_map_ens is None:
          set_ensemble()
            _tmp_map_ens -> current iteration ensemble for mapping
            real_selector.options -> current realisation options iteration
        _tmp_map_idxmap() -> index/name map for current mapping layer and time
        _set_sel_name()
          _sel_name -> current selected (obs) name (may be None)
        highlight_cell() [maybe set _sel_cellid = None]
        update_maphisto_line()
          or
        update_maphisto()
          _histomod()
          update_maphisto_line()

        Parameters
        ----------
        change
        mapfig

        Returns
        -------

        """
        if self._callback_off:
            # if we are in a callback, don't do anything
            return
        # will be used in callback so need to handle change arg
        if mapfig is None:
            mapfig = self.map_widget
        if mapfig is None:
            return
        # get group handler for selected group from outputs dict
        # (these contain ensembles etc)
        gph = self._tmp_map_gph  # is is already set on obs gp change
        assert gph.name == self.map_obs_selector.value, \
            "miss match between cached group and map obs sel"
        # todo: this should trigger select_map_obs_gp?
        # ensemble should already be set on iter/rp radio change
        if self._tmp_map_ens is None:
            self.set_ensemble(change=change, propagate=False)

        ens = self._tmp_map_ens
        # cellids for obs
        idxmap = self._tmp_map_kidxmap # is is already set on layer sel
        t = self._get_tidx(self.map_temporal_slider)
        k = self.layer_selector.value
        if self.reals_or_ptile_radio.value == 'r':
            c = self.real_selector.value
            print(f"Setting map for {gph.name}@k:{k},t:{t},real{c}...")
        else:
            c = int(self.prob_slider.value)
            print(f"Setting map for {gph.name}@k:{k},t:{t},P{c}...")

        # get current selected iteration
        cmap = self.cmap_selector.value
        cr = self.cmap_reverse.value
        try:
            idxmap = idxmap.xs(t, level=-1)
        except KeyError:
            print(f"no index map for group '{gph.name}' @ k:{k}, t:{t}")
            idxmap = pd.Series([])
        self._tmp_map_idxmap = idxmap
        self._set_sel_name()
        try:
            seldf = ens.loc[idxmap.values, c]
            z = seldf.values
            locs = idxmap.index
            meta = locs.get_level_values(0)
        except KeyError:
            z = np.array([])
            locs = np.array([])
            meta = np.array([])
        if len(z) == 0:
            print(f"no map data for group '{gph.name}' @ k:{k}, t:{t}")
        # handle log scale
        if self.map_log_check.value:
            with np.errstate(divide='ignore'):
                z = np.log10(z)
        if cr:
            cmap += '_r'

        if self._uservminmax and len(z) > 0:
            zmin, zmax = self.vminmaxslider.value
            zmin = np.max([zmin, z.min()])
            zmax = np.min([zmax, z.max()])
        else:
            zmin, zmax = [None, None]
        print("vminvmax: ", zmin, zmax)

        with mapfig.batch_update():
            if gph.mapable == 'grid':
                mapfig.update_traces(
                    geojson=self.geojson,
                    z=z,
                    zmin=zmin,
                    zmax=zmax,
                    zauto=True if zmin is None or zmax is None else False,
                    locations=locs,
                    colorscale=cmap,
                    customdata=z,
                    meta=meta,
                    visible=True,
                    selector=dict(name='cpmap')
                )
                mapfig.update_traces(
                    lon=[],
                    lat=[],
                    marker=dict(
                        color=z,
                        colorscale=cmap,
                        cmin=zmin,
                        cmax=zmax,
                        cauto=True if zmin is None or zmax is None else False,
                    ),
                    customdata=z,
                    meta=meta,
                    visible=False,
                    selector=dict(name='scatmap')
                )
                trace = mapfig.data[0]
            else:
                mapfig.update_traces(
                    geojson=[],
                    z=[],
                    zmin=zmin,
                    zmax=zmax,
                    zauto=True if zmin is None or zmax is None else False,
                    locations=locs,
                    colorscale=cmap,
                    customdata=z,
                    meta=meta,
                    visible=False,
                    selector=dict(name='cpmap')
                )
                mapfig.update_traces(
                    lon=locs.get_level_values('x').to_list(),
                    lat=locs.get_level_values('y').to_list(),
                    marker=dict(
                        color=z,
                        colorscale=cmap,
                        cmin=zmin,
                        cmax=zmax,
                        cauto=True if zmin is None or zmax is None else False,
                    ),
                    customdata=z,
                    meta=meta,
                    visible=True,
                    selector=dict(name='scatmap')
                )
                trace = mapfig.data[1]

        if not self._uservminmax:
            with self.callback_off():
                self._reset_vminmax(mapfig=mapfig)
        self.highlight_cell(trace)
        if change is not None:
            if change['owner'] == self.real_selector or change['owner'] == self.prob_slider:
                print("Only updating guide line")
                self.update_maphisto_line()
            else:
                print("Updating histogram")
                self.update_maphisto()
            if change['owner'] == self.map_temporal_slider:
                print("Shifting marker line on tseries")
                self.update_ts_tline()
                if len(self.unmapable) > 0:
                    self.update_ts_tline(self.unmap_ts, self.unmap_temporal_slider)
            elif ((change['owner']==self.real_selector or
                   change['owner']==self.reals_or_ptile_radio or
                   change['owner']==self.iter_selector) and
                  self.reals_or_ptile_radio.value == 'r'):
                print("Updating tseries real selection")
                self.update_ts_real()
                if len(self.unmapable) > 0:
                    self.update_ts_real(self.unmap_ts)
            else:
                print("Updating tseries")
                self.update_ts()


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

    def _get_tidx(self, slider):
        t = slider.options[slider.index]
        if not isinstance(t, str) and len(t) > 1:
            t = t[0]
        return t

    def _set_plotly_unmapfig(self):
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
        unmaphisto.data[-1].update(marker_color='rgba(20,49,220,0.75)')
        unmaphisto.data[0].update(marker_color='rgba(112,112,112,0.75)')
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
        # leave a plotly figure for voila compat
        # convert to figwidget after first update_traces call
        tsplot = go.Figure(layout=dict(margin_t=30,
                                       margin_b=10,
                                       margin_l=10,
                                       margin_r=10,
                                       width=600,
                                       height=300,
                                       title='T-series',
                                       showlegend=True,
                                       xaxis_autorange=True))
        tsplot.add_vline(x=0,
                         line_color='black',
                         line_dash='dash',
                         annotation_text="selected time",
                         annotation_name="selected time",
                         annotation_visible=False,
                         name="tsel",
                         legendgroup="tsel",
                         visible=True,
                         showlegend=True)
        tsplot = go.FigureWidget(tsplot)

        self.unmap_histogram = unmaphisto
        self.unmap_ts = tsplot

    def on_map_click(self, *clickdata):
        trace, p, s = clickdata
        if len(p.point_inds) == 0:
            return
        print("point_inds for sel: ", p.point_inds)
        # get group handler for selected group
        idx = p.point_inds[0]
        print("map index value: ", idx)
        cellid = trace.meta[idx]
        self._sel_cellid = cellid
        self.highlight_cell(trace)
        self._set_sel_name()
        with self.map_histogram.batch_update():
            self.update_maphisto()
        # with self.map_ts.batch_update():
        self.update_ts()

    def highlight_cell(self, trace=None):
        """
        Highlight a specific cell in the map.

        Parameters
        ----------
        trace : plotly.graph_objects.trace.Trace object
            plotly trace object containing the map data.
        """
        if trace is None:
            trace = self.map_widget.data[0]
        mapfig = trace.figure
        cellid = self._sel_cellid
        print("selected cellid :", cellid)
        if hasattr(trace.marker, "line"):
            # base line styles -- Create arrays for line styling
            line_widths = [0.5] * len(trace.meta)
            line_colors = ['gainsboro'] * len(trace.meta)
            with mapfig.batch_update():
                if cellid is not None and cellid in trace.meta:
                    idx = list(trace.meta).index(cellid)
                    # Highlight selected cell
                    print("Highlighting cell:", cellid, "at index", idx)
                    line_widths[idx] = 2
                    line_colors[idx] = 'white'
                else:
                    print("No cell selected or cellid not in map data.")
                    self._sel_cellid = None
                trace.update(marker_line_width=line_widths)
                trace.update(marker_line_color=line_colors)

    def _set_sel_name(self):
        if self._sel_cellid is None:
            idx = None
        else:
            try:
                idx = self._tmp_map_idxmap.xs(self._sel_cellid)
            except KeyError as err:
                # TODO better handling of missed cellid
                print(f"Cell '{self._sel_cellid}' not found in cached index map for group '{self._tmp_map_gph.name}'")
                idx = None
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        self._sel_name = idx

    def _histomod(self, histowgt, df, gp, log=False):
        if df is None:
            histowgt.update_traces(x=[])
            return
        if log:
            df = np.log10(df)
        for i, dfi in df.groupby('iteration'):
            # print(df)
            histowgt.update_traces(x=dfi.values, selector=dict(name=f"iter_{i}"))

        gph = self.obs_gphandlers[gp]
        obsplus = gph.obsplus
        obsidx = df.name
        # if no obsplus or obsidx not in obsplus index, clear obs+noise trace
        if obsplus is None or obsidx not in gph.obsplus.index:
            # no obs+noise for this group
            histowgt.update_traces(x=[], selector=dict(name=f"obs+noise"))
            histowgt.update_traces(x=[None]*50, visible=False,
                                   selector=dict(name=f"obsval"))
            return

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

    def update_maphisto(self):
        # cellid = self._sel_cellid
        idx = self._sel_name
        # gp = self.map_obs_selector.value
        gph = self._tmp_map_gph
        gp = gph.name
        if np.ndim(idx) > 1 and len(idx) > 1:
            warnings.warn("Cellid and tidx match more than one output",
                          UserWarning)
            idx = idx.iloc[0]
        if idx is None:
            self.map_histogram.update_traces(x=[])
            return
        try:
            self._sel_ensdf = self._tmp_map_gph.ens.xs(idx)
        except KeyError as err:
            # TODO better handling of missed idx (obsnme)
            print(f"'{idx}' not found in cached ensemble for group '{gp}'")
            self.map_histogram.update_traces(x=[])
            return
        self._histomod(self.map_histogram, self._sel_ensdf, gp,
                       log=self.map_log_check.value)
        self.update_maphisto_line()

    def update_maphisto_line(self):
        # cellid = self._sel_cellid
        idx = self._sel_name
        if idx is None:
            self.map_histogram.update_traces(x=[] * 50, selector=dict(name=f"mapval"))
            return
        rp = self.reals_or_ptile_radio.value
        if rp == 'r':
            v = self.real_selector.value
        else:
            v = int(self.prob_slider.value)
        data = self._tmp_map_ens.loc[idx, v]
        if self.map_log_check.value and data is not None:
            data = np.log10(data)
        print("Prob/Real value: ", data)
        # Update the vertical line in the histogram
        with self.map_histogram.batch_update():
            # Remove any existing vertical line
            self.map_histogram.update_traces(x=[data] * 50, selector=dict(name=f"mapval"))


    def update_ts(self, v='map'):
        from itertools import cycle
        from plotly.colors import DEFAULT_PLOTLY_COLORS
        if v == 'map':
            tsplot = self.map_ts
            ens = self._tmp_map_gph.ens
            gpname = self._tmp_map_gph.name
            obsplus = self._tmp_map_gph.obsplus
            slider = self.map_temporal_slider
        else:
            tsplot = self.unmap_ts
            ens = self._tmp_unmap_gph.ens
            gpname = self._tmp_unmap_gph.name
            obsplus = self._tmp_unmap_gph.obsplus
            slider = self.unmap_temporal_slider
        #sorted iteration keys for colors on the fly
        iters = sorted(self.real_dict.keys())
        ccycle = cycle(DEFAULT_PLOTLY_COLORS)

        idxs = None
        update = False
        if v == 'map':
            if self._sel_name is None:
                pass  # update stays False
            else:
                # get index across time
                # this may need tweaking for more complex dfs
                try:
                    # get selection mapping from full group infor map
                    # (to account for 'all' layer option)
                    mapdf = self._tmp_map_gph.group_info.ensmap
                    # slicer from selected obsname (across time)
                    try:
                        slicer = mapdf.index[mapdf == self._sel_name].remove_unused_levels().set_levels([slice(None)], level=self.tidx).values[0]
                    except TypeError:  # fails on early pandas versions
                        slicer = mapdf.index[mapdf == self._sel_name].remove_unused_levels()
                        tp = slicer.names.index(self.tidx)
                        slicer = slicer.values[0][:tp] + (slice(None),) + slicer.values[0][tp+1:]
                    # extract indices for selected, across time
                    # again a pandas version variation means that slicer can
                    # return a single series value rather than a series with index
                    try:
                        idxs = mapdf.loc[slicer].sort_index()
                    except AttributeError:
                        idxs = mapdf.loc[[slicer]].sort_index()
                except KeyError as err:  # slicing returns an error (cellid not in map)
                    # TODO better handling of missed idx (obsnme)
                    print(f"'{self._sel_name}' not found in cached "
                          f"ensemble for group '{gpname}'")
        else:
            idxs = self._tmp_unmap_idxmap.sort_index()

        datmode = "markers+lines"  # plot outputs as markers+lines
        if idxs is not None:
            # get time index and obs names -- this will be sorted by time
            x = idxs.index.get_level_values(self.tidx).values
            obsnames = idxs.values.tolist()
            if len(x) > 1:
                # print("obsnames for ts: ", obsnames)
                # print("time idx for ts: ", x)
                # obs across times sliced from ensemble
                df = ens.loc[obsnames, :].set_index(x)
                # print("df for ts: ", df)
                # only trigger update if df is more than one point
                update = True if len(df) > 1 else False

                # also get obs (plus noise) info
                if obsplus is not None:
                    # get obsnames across time
                    obsplus = obsplus.loc[obsnames, :].set_index(x)
                    # if just one unique value, then  we dont have an obs+noise ensemble
                    if (obsplus.nunique(axis=1) == 1).all():
                        obsplus = obsplus.iloc[:, [0]]  # slice out single column
                        obsmode = "markers"  # plot obsvals as markers only
                        datmode = "lines"  # plot outputs as lines only
                        legendgroup = "obsval"  # define legend group for obsval only

                    else:
                        obsmode = "markers+lines"  # plot obs+noise as markers+lines
                        legendgroup = "obs+noise"  # define legend group for obs+noise

        lines = []  # empty unless new data to plot
        if update:
            leg = set([])
            for dfi in df.T.itertuples():
                i, r = dfi[0]  # iteration and real
                if i == iters[0]:
                    c = 'rgba(112,112,112,0.75)'
                elif i == iters[-1]:
                    c = 'rgba(20,49,220,0.75)'
                else:
                    c = next(ccycle)
                lw = 0.7
                lines.append(
                    go.Scattergl(x=x,
                                 y=np.array(dfi[1:]),  # itertuples so 0 is index
                                 name=f'iter_{i}',
                                 meta=f"{i}_{r}",
                                 legendgroup=f'iter_{i}',
                                 mode=datmode,
                                 line=dict(color=c, width=lw),
                                 marker_size=2,
                                 hovertemplate=None,
                                 hoverinfo='none',
                                 opacity=0.5,
                                 showlegend=i not in leg)
                )
                leg.add(i)
            if obsplus is not None:
                leg = True
                for dfi in obsplus.T.itertuples():
                    lines.append(
                        go.Scattergl(x=x,
                                     y=np.array(dfi[1:]),  # itertuples so 0 is index
                                     name=f"obsval_{dfi[0]}",  # need to be this to pick up in update methods
                                     legendgroup=legendgroup,
                                     mode=obsmode,
                                     marker=dict(color='red'),
                                     line=dict(color='red', width=0.5),
                                     hovertemplate=None,
                                     hoverinfo='none',
                                     opacity=0.5,
                                     showlegend=leg)
                    )
                    leg = False  # only show legend once
        self.lines=lines
        # update in batch
        with tsplot.batch_update():
            print(f"Cleaning {v} ts plot")
            # kill all existing data traces except for tsel
            tsplot.data = []
            if len(lines) > 1:
                print(f"Adding updated traces to {v} ts plot")
                tsplot.add_traces(lines)
            self.update_ts_real(tsplot)
            self.update_ts_tline(tsplot, slider)

    def update_ts_real(self, tsplot=None):
        if tsplot is None:
            tsplot = self.map_ts
        with tsplot.batch_update():
            tsplot.update_traces(selector=dict(line_width=10), line_width=0.5)
            if self.reals_or_ptile_radio.value == 'r':
                print("Selecting real in ts...")
                i = self.iter_selector.value
                r = self.real_selector.value
                tsplot.update_traces(selector=dict(meta=f"{i}_{r}"), line_width=10)

    def update_ts_tline(self, tsplot=None, slider=None):
        if tsplot is None:
            tsplot = self.map_ts
        if slider is None:
            slider = self.map_temporal_slider
        t = self._get_tidx(slider)
        # if np.isnan(t):
        #     return
        print("Shifting tslider in ts...")
        with tsplot.batch_update():
            tsplot.update_shapes(
                x0=t,
                x1=t,
                selector=dict(name="tsel")
            )

    def set_unmap_group(self, change=None):
        """
        Call back function triggered by unmap_group_selector changes.
        Propagates downstream to set_unmap_options()
        Parameters
        ----------
        change

        Returns
        -------

        """
        if self._callback_off:
            # if we are in a callback, don't do anything
            return
        # if group selector changes, need to update obs selector options
        # dependent on the selected time (this can get circular!)
        # so we will roll time last up (same as with map)
        print("Setting unmap options...")
        gsel = self.unmap_group_selector.value
        gph = self.obs_gphandlers[gsel]
        # set tmp attibutes for quick lookup later
        self._tmp_unmap_gph = gph
        self._tmp_unmap_gidxmap = gph.group_info.ensmap
        # get current unmap selector value
        osel = self.unmap_selector.value
        opts = self._tmp_unmap_gidxmap.index.unique(0)
        # todo time may or may not be part of this...?
        self.unmap_selector.options=opts
        if osel in self.unmap_selector.options:
            self.unmap_selector.value = osel
        else:
            self.unmap_selector.value = self.unmap_selector.options[0]
        if osel == self.unmap_selector.value:
            self.set_unmap_level(change)

    def set_unmap_level(self, change=None):
        ens = self._tmp_unmap_gph.ens
        l2 = self.unmap_selector.value
        idx = self._tmp_unmap_gidxmap.xs(l2, level=0)
        self._tmp_unmap_idxmap = idx
        ens = ens.loc[idx.values, :]
        self._tmp_unmap_ens = ens
        with self.callback_off():
            self._set_slider_options(self.unmap_temporal_slider,
                                     self._tmp_unmap_gidxmap)
        self.set_unmap(change=change)

    def set_unmap(self, change=None):
        if self._callback_off:
            # if we are in a callback, don't do anything
            return
        # if unmap observation changed need to update histogram
        print("Setting unmap...")
        gsel = self._tmp_unmap_gph.name
        t = self._get_tidx(self.unmap_temporal_slider)  # todo time may or may not be part of this...?
        ens = self._tmp_unmap_ens
        idx = self._tmp_unmap_idxmap.xs(t)
        try:
            seldf = ens.xs(idx)
        except KeyError:
            seldf = pd.Series([])
        with self.unmap_histogram.batch_update():
            self._histomod(self.unmap_histogram, seldf, gsel,
                           log=self.unmap_log_check.value)
        if change is not None and change.owner == self.unmap_temporal_slider:
            print("Shifting marker line on unmap tseries")
            self.update_ts_tline(tsplot=self.unmap_ts,
                                slider=self.unmap_temporal_slider)
        else:
            self.update_ts(v='unmap')


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
                min_width='850px',
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

        ts_box = Box([self.map_ts],
                     layout=Layout(width='100%',
                                   min_height='100px',
                                   display='flex',
                                   justify_content='center',
                                   align_items='center',
                                   # border='3px solid black'
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
            [sel0_box, map_box, slider_box, ts_box],
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
        ts_box = Box([self.unmap_ts],
                     layout=Layout(width='100%',
                                   min_height='100px',
                                   display='flex',
                                   justify_content='center',
                                   align_items='center',
                                   # border='3px solid black'
                                   )
                     )
        unmapbox = ipyw.VBox([
            ipyw.HTML("<h1>Unmappable Obs:</h1>"),
            ipyw.Box([self.unmap_group_selector,
                      self.unmap_selector]),
            ipyw.Box([self.unmap_histogram,
                      ipyw.VBox([self.unmap_log_check,
                                 self.unmap_temporal_slider,
                                 ts_box])])
        ])
        return unmapbox
