#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from scipy.interpolate import RectBivariateSpline
from roost.utils import *


class NormalizedWrappedInterpolant(object):
    """2D interpolant with natural index coordinates

        :var array: 2D array to interpolate
        :type array: array_like

        :var i: normalized coordinates along the first axis
        :type i: array_like

        :var j: normalized coordinates along the second axis
        :type j: array_like

        :var interpolant: wrapped interpolant
        :type interpolant: R² -> R mapping

    """

    def __init__(self, array, interpolant_generator=None):
        """

            :param array: 2D array to interpolate
            :type array: array_like

            :param interpolant_generator: wrapped interpolant
            :type interpolant_generator: R² -> (R² -> R) mapping

            :returns: None

        """

        if interpolant_generator is None:
            interpolant_generator = RectBivariateSpline

        self.shape = array.shape
        self.array = array
        self.i = np.linspace(0, self.shape[0] - 1, self.shape[0])
        self.j = np.linspace(0, self.shape[1] - 1, self.shape[1])
        self.interpolant = interpolant_generator(self.i, self.j, array)

    def __call__(self, *args, **kwargs):
        """

            :param args:
            :type args:

            :param kwargs:
            :type kwargs:

            :returns:

        """

        return self.interpolant(*args, **kwargs)

    def ev(self, *args, **kwargs):
        """

            :param args:
            :type args:

            :param kwargs:
            :type kwargs:

            :returns:

        """

        return self.interpolant.ev(*args, **kwargs)


class TwoFrameInterpolant(object):
    """Performs optical flow or linear interpolation between two consecutive frames

        :var start_frame: Matrix representation of the variable at the start of the temporal interpolation domain
        :type start_frame: 2D array

        :var end_frame: Matrix representation of the variable at the end of the temporal interpolation
        :type end_frame: 2D array

        :var flow_calculator: Callable that computes the optical flow between two frames / arrays
        :type flow_calculator: (R² x R²) -> R² mapping

        :var spatial_interp: Callable that generates a 2D interpolant of a frame array with normalized coordinates
        :type spatial_interp: R² -> (R² -> R)

        :var flow: Optical flow between the start and end frames
        :type flow: 2D array (R²)

    """

    def __init__(self, start_frame, end_frame, flow_calculator=None, spatial_interp=None, initial_flow=None):

        """

            :param start_frame: 2D array representation of the variable at the start of the temporal interpolation domain
            :type start_frame: array_like

            :param end_frame: 2D array representation of the variable at the end of the temporal interpolation
            :type end_frame: array_like

            :param flow_calculator: Callable that computes the optical flow between two frames / arrays
            :type flow_calculator: (R² x R²) -> R² mapping

            :param spatial_interp: Callable that generates a 2D interpolant of a frame array with normalized coordinates
            :type spatial_interp: R² -> (R² -> R)

            :returns: None
        """

        if spatial_interp is None:
            spatial_interp = RectBivariateSpline
        if flow_calculator is None:
            flow_calculator = FarnebackFlow()
        if start_frame.shape != end_frame.shape:
            raise ValueError(f"start_frame and end_frame must have the same shape; instead, they have shapes "
                             f"{start_frame.shape} and {end_frame.shape}")
        if len(start_frame.shape) != 2:
            raise ValueError(f"Input frames with shape {start_frame.shape} are not 2D")

        self.start_frame = start_frame
        self.end_frame = end_frame
        self.shape = self.start_frame.shape
        self.flow_calculator = flow_calculator
        self.spatial_interp = spatial_interp
        self.flow = self.flow_calculator(self.start_frame, self.end_frame, initial_flow)
        r_initial_flow = None if initial_flow is None else -initial_flow
        self.flow_r = - self.flow_calculator(self.end_frame, self.start_frame, r_initial_flow)
        self.I_start = NormalizedWrappedInterpolant(start_frame, spatial_interp)
        self.I_end = NormalizedWrappedInterpolant(end_frame, spatial_interp)
        self.ii, self.jj = np.meshgrid(self.I_start.i, self.I_start.j, indexing='ij')
        self.i_max = self.shape[0] - 1
        self.j_max = self.shape[1] - 1

    def __call__(self, t):

        """
            :param t: Fraction of the time domain between the start and the end frame. Must fulfill 0 <= t <= 1
            :type t: float

            :returns: Flow-interpolated array at t
            :rtype: array_like

        """

        if t < 0.0 or t > 1.0:
            raise ValueError("The temporal fraction for optical flow interpolation must lie between 0 and 1")

        ii_end = np.clip(self.ii + self.flow[:, :, 1] * (1 - t), 0, self.i_max)
        jj_end = np.clip(self.jj + self.flow[:, :, 0] * (1 - t), 0, self.j_max)
        ii_start = np.clip(self.ii - self.flow[:, :, 1] * t, 0, self.i_max)
        jj_start = np.clip(self.jj - self.flow[:, :, 0] * t, 0, self.j_max)

        start_translated = self.I_start(ii_start.flatten(), jj_start.flatten(), grid=False).reshape(self.shape)
        end_translated = self.I_end(ii_end.flatten(), jj_end.flatten(), grid=False).reshape(self.shape)

        return t * end_translated + (1 - t) * start_translated


class MultiFrameInterpolant(object):
    """Performs optical flow or linear interpolation between several consecutive, equally spaced frames

        :var interpolants: List of pairwise interpolants
        :type interpolants: List[TwoFrameInterpolant]

    """
    def __init__(self, frames, flow_calculator=None, spatial_interp=None, t_axis=None, initial_flows=None):
        """

            :param frames: List of 2D arrays
            :type frames: list[array_like]

            :param flow_calculator: Callable that computes the optical flow between two frames / arrays
            :type flow_calculator: (R² x R²) -> R² mapping

            :param spatial_interp: Callable that generates a 2D interpolant of a frame array with normalized coordinates
            :type spatial_interp: R² -> (R² -> R)

            :param t_axis: (optional) array containing the timestamps of t
            :type t_axis: array_like(dtype='timedelta64')

            :param initial_flows: List of arrays of shape N x M x 2 representing initial guesses for the optical flows
            :type initial_flows: list[array_like]

            :returns: None

        """
        assert len(frames) > 1
        if initial_flows is None:
            initial_flows = [None for _ in range(len(frames) - 1)]
        self.frames = frames
        self.t_axis = t_axis
        self.interpolants = [TwoFrameInterpolant(prev_frame, next_frame, flow_calculator, spatial_interp, flow)
                             for prev_frame, next_frame, flow in zip(frames[:-1], frames[1:], initial_flows)]

    def __call__(self, t):

        """

            :param t: Fraction of the time domain between the start and the end frame. Must fulfill 0 <= t <= n_frames
            :type t: float

            :returns: Flow-interpolated array at t
            :rtype: array_like

        """

        if t < 0.0 or t > len(self.frames) - 1:
            raise ValueError("The temporal fraction must lie between 0 and n - 1, where n is the number of frames")
        i = int(t)
        t_frac = t - i
        return self.interpolants[i](t_frac)

    def eval_at_t(self, t, check_nans=True):
        """

            :param t: Union[np.datetime64, np.timedelta64]
            :type t:

            :param check_nans: if True, replace any nans in the input with zeros
            :type check_nans: bool

            :returns:  Flow-interpolated array at timestamp t
            :rtype: array_like

        """

        upper_bound = self.t_axis[-1]
        lower_bound = self.t_axis[0]

        if self.t_axis is None:
            raise ValueError(f"The current {self.__class__} instance was not initialized with a temporal axis")
        if t < lower_bound:
            raise ValueError(f"The timestamp {t} is below the lower temporal axis bound of {lower_bound}")
        if t > upper_bound:
            raise ValueError(f"The timestamp {t} is below the upper temporal axis bound of {upper_bound}")

        idx_low, idx_high = get_bracketing_indexes(self.t_axis, t)
        t_high = self.t_axis[idx_high]
        t_low = self.t_axis[idx_low]
        frac = float(t - t_low) / float(t_high - t_low)
        assert 0 <= frac
        assert 1 >= frac
        output = self.interpolants[idx_low](frac)
        if check_nans:
            assert not np.isnan(output.sum())
        return output


class FarnebackFlow(object):
    """Optical flow calculator using the algorithm of Farneback (2003)

        :var parameters: arguments passed to the Farneback algorithm
        :type parameters: dict


    """

    def __init__(self, **kwargs):
        """Optical flow calculator using the algorithm of Farneback (2003)

            :param kwargs: arguments passed to the Farneback algorithm
            :type kwargs: dict

            :returns: None


        """
        self.parameters = {'iterations': 2,
                           'levels': 9,
                           'poly_n': 10,
                           'poly_sigma': 1.934,
                           'pyr_scale': 0.85,
                           'scale': 25.0,
                           'winsize': 77,
                           'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                           }
        # self.parameters = {'pyr_scale': 0.5,
        #                    'levels': 5,
        #                    'winsize': 100,
        #                    'iterations': 4,
        #                    'poly_n': 9,
        #                    'poly_sigma': 1.6,
        #                    'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        #                    'scale': 1e5}
        self.parameters.update(kwargs)
        for key in ('levels', 'winsize', 'iterations', 'poly_n'):
            self.parameters[key] = int(self.parameters[key])
        self.scale = self.parameters['scale']
        del self.parameters['scale']

    def __call__(self, start_frame, end_frame, initial_flow=None, check_nans=True):
        """Returns the computed optical flow

            :param start_frame: 2D array (N x M) representing the start frame
            :type start_frame: array_like

            :param end_frame: 2D array (N x M) representing the end frame
            :type end_frame: array_like

            :param initial_flow: (optional) N x M x 2 array to initialize the flow
            :type initial_flow: array_like or None

            :param start_frame: 2D array (N x M) representing the start frame
            :type start_frame: array_like

            :param check_nans: (optional) if True, replace any nans in the input with zeros
            :type check_nans: bool

            :returns: optical flow between the frames
            :rtype: array_like


        """
        flow = cv2.calcOpticalFlowFarneback(self.scale * start_frame,
                                            self.scale * end_frame,
                                            initial_flow,
                                            **self.parameters)
        if check_nans:
            assert not np.isnan(flow.sum())
        return flow


class DummyFlow(object):
    def __init__(self):
        pass

    def __call__(self, start_frame, end_frame, initial_flow=None):
        return np.zeros(start_frame.shape + (2,))
