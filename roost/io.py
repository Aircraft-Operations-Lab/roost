# -*- coding: utf-8 -*-

import xarray as xr
import itertools
from roost.optiflow import MultiFrameInterpolant
from roost.utils import *
from math import ceil


def check_array_for_nans(arr):
    """

    :param arr:
    :type arr:

    :returns: None

    """

    assert not np.isnan(arr.sum())


class CoordinateGenerator(object):
    """Helper class for generating coordinates

        :var dict: Coordinates dictionary, mapping axis_name -> np.ndarray

    """

    def __init__(self):
        """This declares the axes variable as a dictionary """

        self.axes = {}

    def add_axis_lims_resolution(self, ax_name, lower, upper, resolution):
        """Add an axis coordinate specifying limits and resolution (i.e. steps between coordinate points)

            :param ax_name: Name of the coordinate axis to be set
            :type ax_name: str

            :param lower: Lower limit of the coordinate axis
            :type lower: float

            :param upper: Upper limit of the coordinate axis
            :type upper: float

            :param resolution: Distance between axis points
            :type resolution: float

            :returns: None

        """
        self.axes[ax_name] = np.arange(lower, upper, resolution)

    def add_axis_lims_n_points(self, ax_name, lower, upper, n, dtype=None):
        """Add an axis coordinate specifying limits and number of points in coordinate axis

            :param ax_name: Name of the coordinate axis to be set
            :type ax_name: str

            :param lower: Lower limit of the coordinate axis
            :type lower: float

            :param upper: Upper limit of the coordinate axis
            :type upper: float

            :param n: Number of points in the coordinate axis
            :type n: int

            :param dtype: The type of the coordinate array. If dtype is not given, it will be inferred by numpy.linspace()
            :type dtype: dtype, optional

            :returns: None

        """
        #self.axes[ax_name] = np.linspace(lower, upper, n, dtype=dtype)
        self.axes[ax_name] = lower + (upper - lower) * np.linspace(0, 1, n)


def get_var_name(dataset, variable):
    """

        :param dataset:
        :type dataset:

        :param variable:
        :type variable:

        :returns: None

    """
    try:
        return dataset[variable].shortname
    except AttributeError:
        return variable


class DatasetHandler(object):
    """Wrapper for a xarray.Dataset containing meteorological information at pressure levels

        :var ds: The dataset
        :type ds: xarray.Dataset


    """

    def __init__(self, ds, output_dtype=np.float32, time_dim='step'):
        """
            :param ds: The dataset
            :type ds: xarray.Dataset

            :param output_dtype: (optional)
            :type output_dtype: data-type

            :param time_dim: (optional)
            :type time_dim: str

            :returns: None

        """
        self.time_dim = time_dim
        self.ds = ds
        self.output_dtype = np.float32
        lat = self.ds.coords['latitude']
        lon = self.ds.coords['longitude']
        self.is_ensemble = True
        try: 
            self.ds.number.values
        except:
            self.is_ensemble = False
        if len(lat.shape) == 1:
            self.geo_grid = 'latlon'
            if lat[0] > lat[1]:
                self.ds = self.ds.reindex(latitude=lat[::-1])
            # Monotonically ascending coordinates are required by some interpolants
        elif len(lat.shape) == 2:
            self.geo_grid = 'xy'
            if lat[0, 0] > lat[1, 0]:
                self.ds = self.ds.reindex(y=self.ds.coords['y'][::-1])
            # Monotonically ascending coordinates are required by some interpolants
        else:
            raise ValueError(f"Cannot handle lat/lon coordinates with shape {lat.shape}")
        self.ds = self.ds.transpose(*self.default_dims)

    @classmethod
    def load_from_steps(cls, path_list, **backend_kwargs):
        """Loads the meteo infromation from multiple grib files with consecutive forecast steps

            :param path_list: List of paths of the grib files
            :type path_list: list

            :param backend_kwargs:
            :type backend_kwargs:

            :returns: cls(ds)

        """
        ds = xr.open_mfdataset(path_list, engine='cfgrib', concat_dim=['step'], combine='nested', **backend_kwargs)
        return cls(ds)

    @property
    def default_dims(self):
        """Returns the default dimensions

            :returns: tuple with the dimensions.

        """
        if self.geo_grid == 'latlon':
            dim_list = ('number', 'level', self.time_dim, 'latitude', 'longitude')
        else:
            dim_list = ('number', 'level', self.time_dim, 'y', 'x')
        return tuple(dim for dim in dim_list if dim in self.ds.dims.keys())

    @property
    def data_variables_list(self):
        """Return a list of variables that are not coordinates"""
        all_vars = set(self.ds.variables.keys())
        coords = set(self.ds.coords)
        data_vars = all_vars - coords
        return data_vars

    def get_geographical_coordinate_slice_by_indexes(self, lat_idx_low, lat_idx_high, lon_idx_low, lon_idx_high):
        """Get a CoordinateGenerator (Not implemented)

            :param lat_idx_low:
            :type lat_idx_low:

            :param lat_idx_high:
            :type lat_idx_high:

            :param lon_idx_low:
            :type lon_idx_low:

            :param lon_idx_high:
            :type lon_idx_high:

            :returns: None

        """

    def complete_new_coords(self, **coords):
        """Generates a coordinate set by completing the specificed new coordinates with the already existing ones


            :param coords: New coordinates, as a dict mapping variable names to monotonically increasing 1D arrays
            :type coords: dict

            :returns: New coordinates, completing missing dimensions with the already existing values
            :rtype: dict

        """
        new_coords = {}
        for ax_name, ax in self.ds.coords.items():
            new_coords[ax_name] = ax
        for ax_name, ax in coords.items():
            new_coords[ax_name] = xr.IndexVariable(ax_name, ax, attrs=self.ds.coords[ax_name].attrs)
        try:
            del new_coords['valid_time']
        except KeyError:
            pass
        return new_coords

    def get_resampled_dataset(self, **coords):
        """Resamples the dataset to new coordinates


            :param coords: A dictionary of new coordinates, mapping coordinate names to numpy arrays. If a coordinate axis is missing, the old coordinate axis is used by default
            :type coords: dict

            :returns: A dataset reinterpolated to the new coordinates
            :rtype: xarray.Dataset

        """
        data_vars = self.data_variables_list
        for dv in data_vars:   # fake loop to pick any element from the data_vars set
            break
        dims = self.ds[dv].dims
        new_coords = self.complete_new_coords(**coords)
        new_shape = tuple(len(new_coords[d]) for d in dims)
        placeholder_dataarray = xr.DataArray(data=np.zeros(new_shape), dims=dims, coords=new_coords)
        return self.ds.interp_like(placeholder_dataarray)

    def get_optical_flow_interpolated_dataset(self, new_coords, variables=None, flow_calculator=None,
                                              spatial_interp=None, verbose=False, zero_out_nans=True):
        """Reinterpolates the dataset to the new coordinates using optical flow interpolation


            :param new_coords: A dictionary of new coordinates, mapping coordinate names to numpy arrays. If a coordinate axis is missing, the old coordinate axis is used by default
            :type new_coords: dict

            :param variables: (optional) List of variables to be transported to the new dataset
            :type variables: list

            :param flow_calculator: (optional) Callable that computes the optical flow between two frames / arrays
            :type flow_calculator: (R² x R²) -> R² mapping

            :param spatial_interp: (optional) Callable that generates a 2D interpolant of a frame array with normalized coordinates
            :type spatial_interp: R² -> (R² -> R)

            :param verbose: (optional) If true, display the progress of the dataset reinterpolation procedure
            :type verbose: bool

            :param zero_out_nans: (optional) if True, replace any nans in the input with zeros
            :type zero_out_nans: bool


            :returns: The reinterpolated dataset
            :rtype: xarray.Dataset


        """
        new_coords = self.complete_new_coords(**new_coords)
        if variables is None:
            variables = [get_var_name(self.ds, var) for var in self.ds.data_vars]
        new_vars = {var: self.get_optical_flow_interpolated_variable(new_coords, var, flow_calculator, spatial_interp,
                                                                     verbose=verbose, zero_out_nans=zero_out_nans)
                    for var in variables}
        return xr.Dataset(data_vars=new_vars, coords=new_coords, attrs=self.ds.attrs)

    @staticmethod
    def get_coords_shape(dims, coords):
        """Returns the shape of an array with the given coordinates and dimension ordering

            :param dims: dimension
            :type dims:

            :param coords: coordinates
            :type coords:


            :returns: The coordinates shape
            :rtype: tuple
        """
        ll_dims = len(coords['latitude'].shape)
        if ll_dims == 1:
            return tuple(coords[d].size for d in dims)
        elif ll_dims == 2:
            geo_coords = ('y', 'x')
            return tuple(coords[d].size for d in dims if d not in geo_coords) + coords['latitude'].shape

    def get_optical_flow_interpolated_variable(self, new_coords, variable,
                                               flow_calculator=None, spatial_interp=None,
                                               flow_padding=.1, verbose=False, zero_out_nans=True):
        """

            :param new_coords: A dictionary of new coordinates, mapping coordinate names to numpy arrays. If a coordinate axis is missing, the old coordinate axis is used by default
            :type new_coords: dict

            :param variable: variable to be transported to the new dataset
            :type variable: str

            :param flow_calculator: (optional) Callable that computes the optical flow between two frames / arrays
            :type flow_calculator: (R² x R²) -> R² mapping

            :param spatial_interp: (optional) Callable that generates a 2D interpolant of a frame array with normalized coordinates
            :type spatial_interp: R² -> (R² -> R)

            :param flow_padding: (optional)
            :type flow_padding: float

            :param verbose: (optional) If true, display the progress of the dataset reinterpolation procedure
            :type verbose: bool

            :param zero_out_nans: (optional) if True, replace any nans in the input with zeros
            :type zero_out_nans: bool


            :returns: The reinterpolated dataset
            :rtype: xarray.Dataset

        """

        reinterpolate = False
        reinterpolate_ens_level = False
        old_coords = self.ds.coords
        intermediate_coords = {}
        if self.time_dim in new_coords:
            intermediate_coords[self.time_dim] = new_coords[self.time_dim]
        limits = {'latitude': (-90.0, 90.0), 'longitude': (-180.0, 180.0)}
        geo_coords = limits.keys()
        for coord in geo_coords:
            try:
                assert np.allclose(new_coords[coord].values, old_coords[coord].values)
            except (ValueError, AssertionError):
                reinterpolate = self.geo_grid

        icg = CoordinateGenerator()
        if reinterpolate == "latlon":
            for coord in geo_coords:
                if coord in new_coords:
                    axis_new = new_coords[coord].values
                    axis_old = old_coords[coord].values
                    resolution = axis_old[1] - axis_old[0]  # assuming regular lat/lon grids
                    span = axis_new[-1] - axis_new[0]
                    padding = flow_padding * span
                    padding_indexes = ceil(padding / resolution)
                    low = max(axis_new[0] - padding_indexes * resolution, limits[coord][0], axis_old.min())
                    high = min(axis_new[-1] + (padding_indexes + 1) * resolution, limits[coord][1], axis_old.max())
                    icg.add_axis_lims_resolution(coord, low, high, resolution)
            intermediate_coords.update(icg.axes)
            reinterpolate_ens_level = True
        elif reinterpolate == "xy":
            raise NotImplementedError("Automatic clipping for optical flow calculation is not implemented for "
                                      "non-Plate-Carrée grids (i.e. 2D lat/lon grids). Regrid before or after"
                                      "optical flow resampling")
        else:
            icg.axes['latitude'] = old_coords['latitude']
            icg.axes['longitude'] = old_coords['longitude']
            for dim in self.default_dims:
                if dim not in (self.time_dim, ):
                    if dim in new_coords.keys():
                        try:
                            assert np.allclose(new_coords[dim].values, old_coords[dim].values)
                        except (ValueError, AssertionError):
                            reinterpolate_ens_level = True
                            break
        intermediate_coords = self.complete_new_coords(**intermediate_coords)
        intermediate_shape = self.get_coords_shape(self.default_dims, intermediate_coords)
        intermediate_var_array = np.zeros(intermediate_shape, dtype=self.output_dtype)
        intermediate_data_array = xr.DataArray(data=intermediate_var_array, coords=intermediate_coords,
                                               dims=self.default_dims, name=variable, attrs=self.ds[variable].attrs)
        coords_to_iterate = {}
        if self.is_ensemble:
            coords_to_iterate_keys = ('number', 'level')
        else:
            coords_to_iterate_keys = ('level',)
        for dim_key in coords_to_iterate_keys:
            try:
                coords_to_iterate[dim_key] = list(new_coords[dim_key].values)
            except KeyError:
                coords_to_iterate[dim_key] = [None]
        iterate_combinations = itertools.product(*coords_to_iterate.values())
        for coords_combi in iterate_combinations:
            if verbose:
                print(coords_combi)
            coords_selector_iter = {key: coords_combi[k] for k, key in enumerate(coords_to_iterate_keys)
                                    if coords_combi[k] is not None}
            coords_selector = coords_selector_iter.copy()
            if reinterpolate == 'latlon':
                coords_selector['latitude'] = icg.axes['latitude']
                coords_selector['longitude'] = icg.axes['longitude']

            def get_values(step, cs=coords_selector):
                _cs = cs.copy()
                _cs[self.time_dim] = step
                return self.ds[variable].sel(**_cs).values

            snapshots = [get_values(step)
                         for step in self.ds.coords[self.time_dim]]
            if zero_out_nans:
                snapshots = list(map(np.nan_to_num, snapshots))
            mfi = MultiFrameInterpolant(snapshots, flow_calculator, spatial_interp, old_coords[self.time_dim].values)
            for i, step in enumerate(new_coords[self.time_dim]):
                frame = mfi.eval_at_t(step.values)
                _csi = coords_selector_iter.copy()
                _csi[self.time_dim] = step
                intermediate_data_array.loc[dict(**_csi)] = frame
        if reinterpolate_ens_level:
            new_shape = self.get_coords_shape(self.default_dims, new_coords)
            new_var_array = np.zeros(new_shape, dtype=self.output_dtype)
            new_data_array = xr.DataArray(data=new_var_array,
                                          coords=new_coords,
                                          dims=self.default_dims,
                                          name=variable,
                                          attrs=self.ds[variable].attrs)
            return intermediate_data_array.interp_like(new_data_array)
        else:
            return intermediate_data_array
