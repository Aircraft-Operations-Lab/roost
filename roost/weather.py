# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage as ndimage
import casadi
import copy

from abc import ABC, abstractmethod
from roost.apm import *

from datetime import datetime
from scipy.signal import convolve2d, convolve

inf = casadi.inf
vertcat = casadi.vertcat


def get_bound_indexes(arr, bounds, verbose=False):
    if verbose:
        print("[gbi] ", bounds, arr)
    if type(arr) == list:
        arr = np.array(arr)
    try:
        assert bounds[0] < bounds[1]
    except AssertionError:
        raise ValueError

    if arr[0] >= bounds[0]:
        low = 0
    else:
        low = np.argmax(arr > bounds[0])
        if low:
            low -= 1

    if arr[-1] <= bounds[-1]:
        high = None
    else:
        high = np.argmin(arr < bounds[1])
        if verbose:
            print(f"high is {high}")
        if high < len(arr):
            high += 1

    return slice(low, high)


def build_2Dt_interpolant(coords, values, name):
    y = values.transpose((0, 2, 1))  # from t, φ, λ to t, λ, φ
    # Since evaluation is (φ, λ, t) we need to give (t, λ, φ) before the flatten
    # to the casadi interpolant (quirk of this function)
    xx = (coords['lat'], coords['lon'], coords['times'])
    return casadi.interpolant(name, 'bspline', xx, y.flatten(), {})


def build_4D_interpolant(coords, values, name):
    y = values.transpose((0, 1, 3, 2))  # from t, P, φ, λ to (t, P, λ, φ)
    # Since evaluation is (φ, λ, P, t) we need to give the array with (t, P, λ, φ) axes before the flatten
    # to the casadi interpolant (quirk of this function)
    cP = [100 * hPa for hPa in coords['levels']]
    xx = (coords['lat'], coords['lon'], cP, coords['times'])
    return casadi.interpolant(name, 'bspline', xx, y.flatten(), {})


class DummyWeather(object):
    def __init__(self, u=0, v=0, T=211, z=11000):
        self.u = u
        self.v = v
        self.T = T
        self.z = z
        self.n_members = 1

    def filter_pl_step(self, *args, **kwargs):
        pass

    def get_interpolants(self):
        wi = {'u': [lambda llt: self.u],
              'v': [lambda llt: self.v],
              'T': [lambda llt: self.T],
              'z': [lambda llt: self.z],
              }
        return wi


class WeatherScenario(ABC):
    @staticmethod
    def interpolant_builder(*args):
        return None

    @abstractmethod
    def u(self, lat, lon, pressure, t):
        pass

    @abstractmethod
    def v(self, lat, lon, pressure, t):
        pass

    @abstractmethod
    def T(self, lat, lon, pressure, t):
        pass

    @abstractmethod
    def H(self, lat, lon, pressure, t):
        pass

    def env_state(self, lat, lon, pressure, t):
        args = (lat, lon, pressure, t)
        uv = (self.u(*args), self.v(*args))
        return EnvironmentState(pressure, self.T(*args), wind_vector=uv)

    @classmethod
    def init_from_arrays(cls, coords, U, V, T, Z):
        arrs_names = zip((U, V, T, Z), ('U', 'V', 'T', 'Z'))
        if Z is None:
            arrs_names = list(arrs_names)[:-1]
        return cls(*[cls.interpolant_builder(coords, arr, name) for arr, name in arrs_names])


class WeatherModel(list):
    def __init__(self, scenarios, bounds):
        self.n_members = len(scenarios)  # number of scenarios
        self.bounds = bounds
        self.extend(scenarios)

    def get_slice_with_first_member(self):
        return self.__class__(self[:1], self.bounds)

    def __repr__(self):
        return f"roc3.weather.WeatherModel with {self.n_members} members and bounds {self.bounds}"
    # def __getattr__(self, idx):
    # return self.scenarios[idx]
    # def __setattr__(self, idx, value):
    # self.scenarios[idx] = value


class WeatherArrays4D(object):

    def __init__(self, axes, u, v, T, Z=None):
        self.axes = axes
        self.u = u
        self.v = v
        self.T = T
        self.Z = Z
        self.n = 1
        if len(self.u.shape) == 5:
            self.n = self.u.shape[2]
            
    @classmethod
    def init_from_dataset(cls, ds):
        return cls(ds.coords, ds['U'], ds['V'], ds['T'])
                
    @classmethod
    def load_from_file(cls, f):
        npz = np.load(f, allow_pickle=True)
        ax_names = ['times', 'levels', 'member', 'lat', 'lon']
        axes = {name: npz[name] for name in ax_names if name in npz}
        return cls(axes, npz['u'], npz['v'], npz['T'], npz['Z'])
    
    def get_average_arrays(self, num_format=np.float32):
        arrs = [self.u, self.v, self.T]
        return [np.mean(arr, axis=2).astype(num_format) for arr in arrs]

    def get_constants_dictionary(self):
        d = {}
        for label, ax_la in self.axes.items():
            ax = np.array(ax_la)
            d[f'n_{label}'] = ax.shape[0]
            d[f'{label}_min'] = ax.min()
            d[f'{label}_max'] = ax.max()
            d[f'{label}_range'] = ax.max() - ax.min()
            d[f'{label}_step'] = (ax.max() - ax.min())/(ax.shape[0] - 1)
        return d

    def save(self, f):
        axarr = {k: np.array(v) for k, v in self.axes.items()}
        np.savez_compressed(f, 
                            u=self.u, v=self.v, T=self.T, Z=self.Z, 
                            **axarr)

class ISAWeather(WeatherScenario):
    def __init__(self):
        self.isa = ISA()
        self.n_members = 1

    def filter_pl_step(self, *args, **kwargs):
        pass

    def u(self, lat, lon, pressure, t):
        return 0

    def v(self, lat, lon, pressure, t):
        return 0

    def T(self, lat, lon, pressure, t):
        return self.isa.IT(H2h(P2Hp(pressure)))

    def H(self, lat, lon, pressure, t):
        return H2h(P2Hp(pressure))


class WeatherScenario2Dt(WeatherScenario):
    interpolant_builder = build_2Dt_interpolant

    def __init__(self, Iu, Iv, IT, IZ=None):
        self.Iu = Iu
        self.Iv = Iv
        self.IT = IT
        self.IZ = IZ

    # @classmethod
    # def init_from_arrays(cls, coords, U, V, T, Z):
    #     arrs_names = zip((U, V, T, Z), ('U', 'V', 'T', 'Z'))
    #     if Z is None:
    #         arrs_names = list(arrs_names)[:-1]
    #     return cls(*[build_2Dt_interpolant(coords, arr, name) for arr, name in arrs_names])
    def u(self, lat, lon, pressure, t):
        return self.Iu(vertcat(lat, lon, t))

    def v(self, lat, lon, pressure, t):
        return self.Iv(vertcat(lat, lon, t))

    def T(self, lat, lon, pressure, t):
        return self.IT(vertcat(lat, lon, t))

    def H(self, lat, lon, pressure, t):
        if self.IZ is None:
            return P2Hp(pressure)
        else:
            return self.IZ(vertcat(lat, lon, t))


class WeatherScenario4D(WeatherScenario):
    interpolant_builder = build_4D_interpolant

    def __init__(self, Iu, Iv, IT, IZ=None):
        self.Iu = Iu
        self.Iv = Iv
        self.IT = IT
        self.IZ = IZ

    # @classmethod
    # def init_from_arrays(cls, coords, U, V, T, Z):
    #     arrs_names = zip((U, V, T, Z), ('U', 'V', 'T', 'Z'))
    #     if Z is None:
    #         arrs_names = list(arrs_names)[:-1]
    #     return cls(*[build_4D_interpolant(coords, arr, name) for arr, name in arrs_names])
    def u(self, lat, lon, pressure, t):
        return self.Iu(vertcat(lat, lon, pressure, t))

    def v(self, lat, lon, pressure, t):
        return self.Iv(vertcat(lat, lon, pressure, t))

    def T(self, lat, lon, pressure, t):
        return self.IT(vertcat(lat, lon, pressure, t))

    def H(self, lat, lon, pressure, t):
        if self.IZ is None:
            return P2Hp(pressure)
        else:
            return self.IZ(vertcat(lat, lon, pressure, t))


class GeoArrayHandler(object):
    bitriangular_filter = np.array([[.0625, .125, .0625], [.125, .25, .125], [.0625, .125, .0625]])
    triangular_filter = np.array([.25, .5, .25])
    axes: dict

    def get_coords(self):
        new_axes = {}
        for name, ax in self.axes.items():
            if not self.cfg['predecimate'] and name in ('lat', 'lon'):
                new_axes[name] = ax[::2 ** self.downsample_steps]
            else:
                new_axes[name] = ax[:]
        return new_axes

    def decimate(self, array):
        array = array.astype(self.cfg['format'])
        for i in range(self.downsample_steps):
            array = self.down2(array)
        return array

    def decimate_3d(self, array):
        for i in range(array.shape[0]):
            arr2 = self.decimate(array[i, :, :])
            nlat, nlon = arr2.shape
            array[i, :nlat, :nlon] = arr2
        return array[:, :nlat, :nlon]

    def decimate_4d(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                arr2 = self.decimate(array[i, j, :, :])
                nlat, nlon = arr2.shape
                array[i, j, :nlat, :nlon] = arr2
        return array[:, :, :nlat, :nlon]

    def decimate_5d(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    arr2 = self.decimate(array[i, j, k, :, :])
                    nlat, nlon = arr2.shape
                    array[i, j, k, :nlat, :nlon] = arr2
        return array[:, :, :, :nlat, :nlon]

    @classmethod
    def down2_coord(cls, array):
        return array[::2]

    def down2(self, array):
        '''
        Decimates a 2D array by a factor of two after applying a triangular filter
        '''
        if len(array.shape) != 2:
            raise ValueError(f"Array to decimate must be 2D. Input has shape {array.shape}")
        filtered = convolve2d(array, self.bitriangular_filter, boundary='symm', mode=self.cfg['convolution_mode'])
        return filtered[::2, ::2]


class WeatherStore(GeoArrayHandler):
    def __init__(self):
        pass


class WeatherStore_2Dt_level(WeatherStore):
    def __init__(self, path, level, flipud='auto', **weather_config):
        # values array axes: times, levels, members, lats, lons
        self.cfg = {
            'format': np.float32,
            'downsample_format': np.float16,
            'll_resolution': 1.0,
            'convolution_mode': 'same',
            'skip_geopotential': True,
            'predecimate': False,
            'time_offset': 0.
        }
        self.path = path
        self.cfg.update(weather_config)
        if self.cfg['skip_geopotential']:
            self.variable_names = ['U', 'V', 'T']
        else:
            self.variable_names = ['U', 'V', 'T', 'Z']
        self.npz = np.load(path, allow_pickle=True)
        self.axes = {}
        self.axes['lat'] = self.npz['U'].item()['lats']
        if flipud == 'auto':
            flipud = self.axes['lat'][1] < self.axes['lat'][0]
        if flipud:
            self.axes['lat'] = self.axes['lat'][::-1]
        self.axes['lon'] = self.npz['U'].item()['longs']
        self.npz_resolution = self.axes['lat'][1] - self.axes['lat'][0]
        self.axes['levels'] = self.npz['U'].item()['levels']
        self.axes['times'] = self.npz['U'].item()['times']+self.cfg["time_offset"]
        self.n_members = self.npz['U'].item()['values'].shape[2]
        self.members = range(self.n_members)
        self.downsample_steps = int(np.log2(self.cfg['ll_resolution'] // self.npz_resolution))
        levels = list(self.axes['levels'])
        try:
            i_lvl = levels.index(level)
        except ValueError:
            raise ValueError(f"Could not find data for the pressure level {level} hPa in the file {self.path}; available levels are: {levels}")
        self.values = {}
        for tag in self.variable_names:
            if flipud:
                self.values[tag] = self.npz[tag].item()['values'][:, i_lvl, :, ::-1, :]
            else:
                self.values[tag] = self.npz[tag].item()['values'][:, i_lvl, :, :, :]
            # self.axes['levels'] = self.axes['levels']

    def reduce_domain(self, bounds):
        slice_idx = {}
        if 'times' in bounds:
            bounds['times'] = list(bounds['times'])
            for i in (0, 1):
                bti = bounds['times'][i]
                if type(bti) == datetime:
                    bounds['times'][i] = bti.timestamp()
        else:
            slice_idx['times'] = slice(0, None)
        for ax_name, ax_bounds in bounds.items():
            try:
                slc = get_bound_indexes(self.axes[ax_name], ax_bounds)
            except ValueError:
                print(f"Could not reduce domain along the '{ax_name}' axis")
                print(f"Current bounds: ({self.axes[ax_name][0]}, {self.axes[ax_name][-1]})")
                print(f"Desired bounds: {ax_bounds}")
                raise
            slice_idx[ax_name] = slc
            self.axes[ax_name] = self.axes[ax_name][slc]
        for tag in self.variable_names:
            self.values[tag] = self.values[tag][slice_idx['times'],
                               :,  # get all members
                               slice_idx['lat'],
                               slice_idx['lon']]

    def get_weather_model(self, n_members=casadi.inf):
        coords = self.get_coords()
        scenarios = []
        n_members = min(self.n_members, n_members)
        for i in range(n_members):
            variable_arrays = []
            for v in self.variable_names:
                arr = self.decimate_3d(self.values[v][:, i, :, :])
                variable_arrays.append(arr)
            if 'Z' not in self.variable_names:
                variable_arrays.append(None)
            scenario = WeatherScenario2Dt.init_from_arrays(coords, *variable_arrays)
            scenarios.append(scenario)
        bounds = {}
        for name, axis in coords.items():
            bounds[name] = (min(axis), max(axis))
        return WeatherModel(scenarios, bounds)


class WeatherScenario2D(WeatherScenario):
    def __init__(self, Iu, Iv, IT, IZ):
        self.Iu = Iu
        self.Iv = Iv
        self.IT = IT
        self.IZ = IZ

    def u(self, lat, lon, pressure, t):
        return self.Iu(vertcat(lat, lon))

    def v(self, lat, lon, pressure, t):
        return self.Iv(vertcat(lat, lon))

    def T(self, lat, lon, pressure, t):
        return self.IT(vertcat(lat, lon))

    def H(self, lat, lon, pressure, t):
        return self.IZ(vertcat(lat, lon))


class WeatherGribs2D(object):
    def __init__(self, gribs, tref=None):
        self.gribs = gribs
        self.n_members = 1
        # In this instance, tref doesn't matter, but with 4D weather it will

    def filter_pl_step(self, pl, fcst_step=6, fcst_time=0, maxn=101, regrid=1,
                       debug_mode=False, **kwargs):
        """
        - pl: hPa
        - fcst_step: h
        """
        u_names = ['u-component of wind', 'U component of wind', 'U velocity']
        v_names = ['v-component of wind', 'V component of wind', 'V velocity']
        T_names = ['Temperature']
        z_names = ['Geopotential Height', 'Geopotential height', 'Geopotential']
        d_vars = {
            0: u_names,
            1: v_names,
            2: T_names,
            3: z_names,
        }

        self.vcodes = {
            0: 'u',
            1: 'v',
            2: 'T',
            3: 'z',
        }
        grb0 = self.gribs[1]
        ll = grb0.latlons()
        if ll[0][0, 0] > ll[0][1, 0]:
            flip = True
        else:
            flip = False
        if flip:
            self.lats2d = np.flipud(ll[0])[::regrid, ::regrid]
            self.lons2d = np.flipud(ll[1])[::regrid, ::regrid]
        else:
            self.lats2d = ll[0][::regrid, ::regrid]
            self.lons2d = ll[1][::regrid, ::regrid]
        self.gribs.rewind()
        arr = np.zeros((maxn, 4, self.lats2d.shape[0], self.lats2d.shape[1]), dtype=np.float64)  # ens_n, lat, lon
        n_members = 1
        processed_grids = 0

        for grb in self.gribs:
            idx = None
            if fcst_step is not None and grb.forecastTime != fcst_step:
                if debug_mode:
                    print("+ Discarded grib -------")
                    print("{0}".format(grb))
                    print("fcst_step: desired {0}, found {1}".format(fcst_step, grb.forecastTime))
                continue
            if grb.dataTime != fcst_time:
                if debug_mode:
                    print("+ Discarded grib -------")
                    print("{0}".format(grb))
                    print("fcst_time: desired {0}, found {1}".format(fcst_time, grb.dataTime))
                continue
            if grb.level != pl:
                if debug_mode:
                    print("+ Discarded grib -------")
                    print("{0}".format(grb))
                    print("pressure_level: desired {0}, found {1}".format(pl, grb.level))
                continue
            processed_grids += 1
            for key, names in d_vars.items():
                if grb.parameterName in names:
                    idx = key
            if idx is not None:
                if grb.perturbationNumber == 0:
                    member = 0  # check this logic? same as pertNumber == 1
                else:
                    member = grb.perturbationNumber - 1
                if member < maxn:
                    if flip:
                        arr[member, idx, :, :] = np.flipud(grb.values[::regrid, ::regrid])
                    else:
                        arr[member, idx, :, :] = grb.values[::regrid, ::regrid]
            n_members = min(max(n_members, grb.perturbationNumber), maxn)
        assert processed_grids > 0
        self.n_members = n_members
        self.wa = arr[:self.n_members, :, :, :]
        latlist = self.lats2d[:, 0]
        lonlist = self.lons2d[0, :]
        x = [latlist, lonlist]
        self.latlist = latlist
        self.lonlist = lonlist

    def get_weather_model(self):
        scenarios = []
        wi = weather_interpolants = self.get_interpolants()
        for i in range(self.n_members):
            scenario = WeatherScenario2D(wi['u'][i], wi['v'][i], wi['T'][i], wi['z'][i])
            scenarios.append(scenario)
        bounds = {}
        bounds['lat'] = (min(self.latlist), max(self.latlist))
        bounds['lon'] = (min(self.lonlist), max(self.lonlist))
        return WeatherModel(scenarios, bounds)

    def get_stdev(self):
        u_mean = self.wa[:, 0, :, :].mean(axis=0)
        v_mean = self.wa[:, 1, :, :].mean(axis=0)
        u_diff = self.wa[:, 0, :, :] - u_mean
        v_diff = self.wa[:, 0, :, :] - u_mean
        u_devs_sq = (u_diff ** 2).mean(axis=0)
        v_devs_sq = (v_diff ** 2).mean(axis=0)
        w_devs_sq = u_devs_sq + v_devs_sq
        return w_devs_sq ** 0.5

    def get_interpolants(self, cutoff=inf, kind='casadi', casadi_itype='bspline'):
        wi = {'u': [],
              'v': [],
              'T': [],
              'z': [],
              }
        latlist = self.lats2d[:, 0]
        lonlist = self.lons2d[0, :]
        x = [latlist, lonlist]
        self.latlist = latlist
        self.lonlist = lonlist
        N = min(self.n_members, cutoff)
        for i in range(N):
            for idx, vname in self.vcodes.items():
                if kind == 'casadi':
                    # Transpose, since casadi interp inverts arg order ¯\_(ツ)_/¯
                    y = self.wa[i, idx, :, :].T.flatten()
                    I = casadi.interpolant(vname + str(i), casadi_itype, x, y, {})
                elif kind == 'scipy':
                    y = self.wa[i, idx, :, :]
                    I = scipy.interpolate.RectBivariateSpline(latlist, lonlist, y)
                wi[vname].append(I)
        return wi


# class ArraySlicer(object):
#     def __init__(self, array, axes_list):
#         self.array = array
#         self.axes_list = axes_list
#     def __getattr__(self, )


class WeatherStore_4D(WeatherStore):

    def __init__(self, path, flipud='auto', **weather_config):
        # values array axes: times, levels, members, lats, lons
        self.path = path
        self.cfg = {
            'format': np.float32,
            'downsample_format': np.float16,
            'll_resolution': 1.0,
            'convolution_mode': 'same',
            'skip_geopotential': True,
            'predecimate': True,
            'time_offset': 0.,
        }
        self.cfg.update(weather_config)
        if self.cfg['skip_geopotential']:
            self.variable_names = ['U', 'V', 'T']
        else:
            self.variable_names = ['U', 'V', 'T', 'Z']
        self.npz = np.load(path, allow_pickle=True)
        self.axes = {}
        self.axes['lat'] = self.npz['U'].item()['lats']
        if flipud == 'auto':
            flipud = self.axes['lat'][1] < self.axes['lat'][0]
        if flipud:
            self.axes['lat'] = self.axes['lat'][::-1]
        self.axes['lon'] = self.npz['U'].item()['longs']
        self.npz_resolution = self.axes['lat'][1] - self.axes['lat'][0]
        self.axes['levels'] = list(self.npz['U'].item()['levels'])
        self.axes['times'] = self.npz['U'].item()['times']+self.cfg["time_offset"]
        self.n_members = self.npz['U'].item()['values'].shape[2]
        self.members = range(self.n_members)
        if self.cfg['ll_resolution'] == self.npz_resolution:
            self.downsample_steps = 0
        else:
            self.downsample_steps = int(np.log2(self.cfg['ll_resolution'] // self.npz_resolution))
        self.values = {}
        if 'level' in self.cfg:
            if self.cfg['level'] not in self.axes['levels']:
                raise ValueError
        for tag in self.variable_names:
            if flipud:
                A = self.npz[tag].item()['values'][:, :, :, ::-1, :].astype(self.cfg['format'])
            else:
                A = self.npz[tag].item()['values'][:, :, :, :, :].astype(self.cfg['format'])
            if self.cfg['predecimate']:
                A = self.decimate_5d(A)
            self.values[tag] = A
        if self.cfg['predecimate']:
            self.axes['lat'] = self.axes['lat'][::2 ** self.downsample_steps]
            self.axes['lon'] = self.axes['lon'][::2 ** self.downsample_steps]

    def reduce_domain(self, bounds, verbose=False):
        slice_idx = {}
        if 'times' in bounds:
            bounds['times'] = list(bounds['times'])
            for i in (0, 1):
                bti = bounds['times'][i]
                if type(bti) == datetime:
                    bounds['times'][i] = bti.timestamp()
        else:
            slice_idx['times'] = slice(0, len(self.axes['times']))
        for ax_name, ax_bounds in bounds.items():
            try:
                slc = get_bound_indexes(self.axes[ax_name], ax_bounds)
            except ValueError:
                print(f"Could not reduce domain along the '{ax_name}' axis")
                print(f"Current bounds: ({self.axes[ax_name][0]}, {self.axes[ax_name][-1]})")
                print(f"Desired bounds: {ax_bounds}")
                raise
            slice_idx[ax_name] = slc
            self.axes[ax_name] = self.axes[ax_name][slc]
        for tag in self.variable_names:
            if verbose:
                print("Reducing domain from shape: ", self.values[tag].shape)
            self.values[tag] = self.values[tag][slice_idx['times'],
                                                :,  # get all members
                                                :,  # get all levels
                                                slice_idx['lat'],
                                                slice_idx['lon']]
            if verbose:
                print("Reducing domain to   shape: ", self.values[tag].shape)
                verbose = False

    def get_weather_model(self, n_members=casadi.inf):
        coords = self.get_coords()
        scenarios = []
        n_members = min(self.n_members, n_members)
        for i in range(n_members):
            variable_arrays = []
            for v in self.variable_names:
                arr = self.values[v][:, :, i, :, :]
                if not self.cfg['predecimate']:
                    arr = self.decimate_4d(arr)
                variable_arrays.append(arr)
            if 'Z' not in self.variable_names:
                variable_arrays.append(None)
            scenario = WeatherScenario4D.init_from_arrays(coords, *variable_arrays)
            scenarios.append(scenario)
        bounds = {}
        for name, axis in coords.items():
            bounds[name] = (min(axis), max(axis))
        return WeatherModel(scenarios, bounds)

    def get_weather_arrays(self, n_members=casadi.inf):
        variable_arrays = []
        n_members = min(self.n_members, n_members)
        i_lvl_50 = self.axes['levels'].index(50)
        i_lvl_300 = self.axes['levels'].index(300)
        for v in self.variable_names:
            arr = self.values[v][:, i_lvl_50:(i_lvl_300 + 1), :n_members, :, :]
            if not self.cfg['predecimate']:
                arr = self.decimate_5d(arr)
            variable_arrays.append(arr)
        if 'Z' not in self.variable_names:
            variable_arrays.append(None)
        va = variable_arrays
        new_axes = copy.deepcopy(self.axes)
        new_axes['levels'] = new_axes['levels'][i_lvl_50:(i_lvl_300 + 1)]
        return WeatherArrays4D(new_axes, *va)

    @property
    def ax_names(self):
        return ['times', 'levels', 'member', 'lat', 'lon']

    @staticmethod
    def sort_axes(axes):
        axes_sorted = {}
        for k in self.ax_names:
            axes_sorted[k] = axes[k]
        return axes_sorted

    def as_xarray_dataset(self):
        import xarray as xr
        axes = copy.deepcopy(self.axes)
        axes['member'] = np.array(range(self.n_members))
        variables = {}
        for var, values in self.values.items():
            variables[var] = xr.DataArray(values, 
                                          dims=self.ax_names,
                                          coords=[axes[a] for a in self.ax_names])
        return xr.Dataset(data_vars=variables)

    def get_weather_arrays_resampled(self, shape_lat_lon_t=(32, 32, 16), n_members=casadi.inf):
        import xarray as xr
        new_axes = {}
        for ax_name, ax_length in zip(('lat', 'lon', 'times'), shape_lat_lon_t):
            new_axes[ax_name] = np.linspace(self.axes[ax_name][0], self.axes[ax_name][-1], ax_length)
        new_axes['levels'] = [50, 200, 250, 300]  # hardcoded
        n_members = min(self.n_members, n_members)
        new_axes['member'] = np.array(range(n_members))
        ds_in = self.as_xarray_dataset()
        new_shape = [len(new_axes[a]) for a in self.ax_names]
        placeholder_dataarray = xr.DataArray(data=np.zeros(new_shape), 
                                             dims=self.ax_names,
                                             coords=[new_axes[a] for a in self.ax_names])
        return WeatherArrays4D.init_from_dataset(ds_in.interp_like(placeholder_dataarray))

