#!/usr/bin/python
# -*- coding: utf-8 -*-

from roost.constants import *
from casadi import interpolant


class CasadiISA(object):
    """
    Casadi implementation of the ISA atmosphere (troposphere and tropopause only)

        :var deltaT: Temperature differential w.r.t. ISA conditions
        :type deltaT: float

    """

    def __init__(self, deltaT=0):
        """

            :param deltaT: Temperature differential w.r.t. ISA conditions
            :type deltaT: float, optional

            :returns: None

        """
        self.deltaT = deltaT

        z = np.arange(0, 20.5e3, 500)
        T = np.maximum(T0_MSL + lapse_rate_troposphere * z, T_tropopause * np.ones_like(z))

        P_tropo = P0_MSL * (T / T0_MSL) ** (-g / Rg / lapse_rate_troposphere)
        P_pause = P_tropo[22] * np.exp(-g / Rg / T * (z - h_tropopause))
        P = np.hstack([P_tropo[:23], P_pause[23:]])
        z_I = np.arange(0, 20.5e3, 341)
        T_I = np.maximum(T0_MSL + lapse_rate_troposphere * z_I, T_tropopause * np.ones_like(z_I))

        self.IP = interpolant('P', 'bspline', [z], P, {})
        self.IT = interpolant('T', 'bspline', [z_I], T_I, {})

