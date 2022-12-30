#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# Unless specified otherwise, all units are SI

#: :obj:`float` :
#: kappa = 1.4
kappa = 1.4

#: :obj:`float` :
#: Rg = 287.05287
Rg = 287.05287

#: :obj:`float` :
#: ft = 0.3048
ft = 0.3048

#: :obj:`float` :
#: m2ft = ft**-1
m2ft = ft**-1

#: :obj:`float` :
#: T0_MSL = 288.15
T0_MSL = 288.15

#: :obj:`float` :
#: P0_MSL = 101325.0
P0_MSL = 101325.0

#: :obj:`float` :
#: rho0_MSL = 1.225
rho0_MSL = 1.225

#: :obj:`float` :
#: a0_MSL = 340.294
a0_MSL = 340.294

#: :obj:`float` :
#: nmi = 1852.0
nmi = 1852.0

#: :obj:`float` :
#: kt = nmi/3600
kt = nmi/3600

#: :obj:`float` :
#: ms2kt = 1/kt
ms2kt = 1/kt

#: :obj:`float` :
#: g0 = g = 9.80665
g0 = g = 9.80665

#: :obj:`float` :
#: lapse_rate_troposphere = -0.0065
lapse_rate_troposphere = -0.0065

#: :obj:`float` :
#: R_mean = 6356.8e3
R_mean = 6356.8e3

#: :obj:`float` :
#: d2r = np.pi/180
d2r = np.pi/180

#: :obj:`float` :
#: flattening = 1/298.257223563
flattening = 1/298.257223563

#: :obj:`float` :
#: first_eccentricity = (flattening*(2 - flattening))**0.5
first_eccentricity = (flattening*(2 - flattening))**0.5

#: :obj:`float` :
#: R_a = 6378.137e3
R_a = 6378.137e3

#: :obj:`float` :
#: R_b = 6356.752
R_b = 6356.752

#: :obj:`float` :
#: h_tropopause = 11e3
h_tropopause = 11e3

#: :obj:`float` :
#: T_tropopause = 216.65
T_tropopause = 216.65

#: :obj:`float` :
#: minute = np.timedelta64(1, 'm').astype('timedelta64[ns]')
minute = np.timedelta64(1, 'm').astype('timedelta64[ns]')

#: :obj:`float` :
#: hour = np.timedelta64(1, 'h').astype('timedelta64[ns]')
hour = np.timedelta64(1, 'h').astype('timedelta64[ns]')
