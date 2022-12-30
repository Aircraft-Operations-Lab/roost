import xarray as xr
import numpy as np
from roost.io import CoordinateGenerator, DatasetHandler, get_var_name
from .cudatools import *
import pycuda.driver as drv
import datetime as dt

BASELINE_ORDER = ('t', 'u', 'v', 'z', 'r', 'pv', 'C1', 'C2', 'olr', 'aCCF_CH4')



def var_name_key(var_name, baseline_order=BASELINE_ORDER):
    for i, candidate_name in enumerate(baseline_order):
        if var_name == candidate_name:
            return i
    return len(baseline_order)


class CoordinatesFromLimits(CoordinateGenerator):

    def __init__(self, limits, shape=None):
        super(CoordinatesFromLimits, self).__init__()
        _shape = {
            'latitude': 32,
            'longitude': 32,
            'time': 4,
        }
        if shape is None:
            shape = {}
        _shape.update(shape)
        shape = _shape

        _limits = {'isobaricInhPa': [ 50,  70,  100,  125,  150,  175,  200,  225,  250,  300,  350, 400,  450,  500,  550,  600,  700,  800,  900, 1000]}
        _limits.update(limits)
        limits = _limits

        for var, lim in limits.items():
            if var == 'isobaricInhPa':
                self.axes['isobaricInhPa'] = lim
            else:
                self.add_axis_lims_n_points(var, lim[0], lim[1], shape[var])


class WeatherCache:

    def __init__(self, cache_path, source_dataset, new_coords=None, offsets=None, var_filter=None):

        self.textures = {}
        self.path = cache_path
        if offsets is None:
            offsets = {}
        _offsets = {'t': -150,
                    'u': 200,
                    'v': 200,
                    'z': 0,
                    'r': 0.5,
                    'pv': 1e-4,
                    'C1': 0,
                    'C2': 0,
                    'olr': 0,
                    'aCCF_CH4': -3e-13
                    }
        _offsets.update(offsets)
        self.offsets = _offsets
        if var_filter is None:
            var_filter = BASELINE_ORDER
        try:
            self.ds = xr.open_dataset(self.path, engine='h5netcdf')
        except (FileNotFoundError, OSError):
            dsh = DatasetHandler(source_dataset, time_dim='time')
            self.ds = dsh.get_optical_flow_interpolated_dataset(new_coords)
            var_names = [get_var_name(self.ds, var) for var in self.ds.data_vars]
            encoding = {var: {'compression': 'lzf', 'dtype': np.float32} for var in var_names}
            self.ds.to_netcdf(cache_path, engine='h5netcdf', encoding=encoding)

        candidate_vars = [get_var_name(self.ds, var) for var in self.ds.data_vars if var in var_filter]
        self.var_names = sorted(candidate_vars, key=var_name_key)
        for v in self.var_names:
            if v not in self.offsets.keys():
                self.offsets[v] = 0
        self.n_vars = len(self.var_names)
        self.n_cubes = int(np.ceil(self.n_vars / 4))
        self.cubes = [self.var_names[4 * i: 4 * (i + 1)] for i in range(self.n_cubes)]
        self.n_members = len(self.ds.coords['number'])
        self.n_lat = len(self.ds.coords['latitude'])
        self.n_lon = len(self.ds.coords['longitude'])
        self.n_levels = len(self.ds.coords['isobaricInhPa'])
        self.n_times = len(self.ds.coords['time'].values)
        levels = self.ds.coords['isobaricInhPa'].values
        self.P_triplets = [(P_low, P_high, P_high - P_low) for P_low, P_high in zip(levels[:-1], levels[1:])]
        self.cube_shape = (self.n_times * self.n_levels, 4, self.n_lat, self.n_lon)

    @property
    def start_time(self):
        return self.ds.coords['time'].values[0]

    def get_constants_dictionary(self):
        d = {'n_members': self.n_members,
             'n_cubes': self.n_cubes,
             'levels_hpa': list(self.ds.coords['isobaricInhPa'].values),
             'P_triplets': self.P_triplets,
             'n_times': self.n_times,
             'n_levels': self.n_levels,
             'wx_offsets': []}
        for i, cube in enumerate(self.cubes):
            offsets = [self.offsets[var_name] for var_name in cube]
            while len(offsets) < 4:
                offsets.append(0.0)
            d[f'wx_offsets'].append(offsets)
        for label, ax_la in self.ds.coords.items():
            ax = ax_la.values
            if label == 'time':
                epoch = np.datetime64('1970-01-01T00:00:00Z')
                ax = np.array([(value - epoch) / np.timedelta64(1, 's') for value in ax])
            d[f'n_{label}'] = ax.shape[0]
            d[f'{label}_min'] = ax.min()
            d[f'{label}_max'] = ax.max()
            d[f'{label}_range'] = ax.max() - ax.min()
            d[f'{label}_step'] = (ax.max() - ax.min()) / (ax.shape[0] - 1)
        return d

    def bind_textures_to_module(self, module):
        from pathlib import Path
        from roost.weather import WeatherArrays4D

        for cube_idx, cube in enumerate(self.cubes):
            for member_idx in range(self.n_members):
                ensemble_number = self.ds.coords['number'].values[member_idx]
                arr = np.zeros(shape=self.cube_shape, dtype=np.float32)
                for var_idx, var in enumerate(cube):
                    for P_idx, P in enumerate(self.ds.coords['isobaricInhPa'].values):
                        low = P_idx * self.n_times
                        high = (P_idx + 1) * self.n_times
                        arr[low:high, var_idx, :, :] = self.ds[var].sel(isobaricInhPa=P, number=ensemble_number) + \
                                                       self.offsets[var]
                        # The offset seeks to move the variables to a positive range
                        # The reason is that, apparently, the CUDA trilinear filter for 3D texture fetches
                        # may overflow when interpolating between values of different sign.
                        # Interpolating between same sign values (thanks to the offset)
                        # represents, thus, a useful workaround.
                tex_array = flip_weather_array_for_texture(arr)
                cuda_array = np_to_array_4d_f4(tex_array)
                tex_name = f'wxcube_{cube_idx}_{member_idx}'
                texref = module.get_texref(tex_name)
                self.textures[tex_name] = texref
                texref.set_array(cuda_array)
                texref.set_flags(drv.TRSF_NORMALIZED_COORDINATES)
                texref.set_filter_mode(drv.filter_mode.LINEAR)  # LINEAR)
                texref.set_address_mode(0, drv.address_mode.CLAMP)
                texref.set_address_mode(1, drv.address_mode.CLAMP)
                texref.set_address_mode(2, drv.address_mode.CLAMP)
                
