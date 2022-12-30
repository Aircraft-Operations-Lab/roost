#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def get_bracketing_indexes(array, value):
    """Returns consecutive indexes idx_low and idx_high such that array[idx_low] <= value < array[idx_high].

    :param array: Array to search. Must be 1D and monotonically sorted in ascending order. This condition is NOT checked here
    :type array: array_like

    :param value: Value to bracket
    :type value: Union[float, np.datetime64, np.timedelta64]

    :returns: Indexes that fulfill the condition array[idx_low] <= value < array[idx_high].
    :rtype: tuple

    """

    i_high = np.argmin(array <= value)
    i_low = i_high - 1
    return i_low, i_high
