#!/usr/bin/env python3

from numpy import sin, cos, arcsin, arctan2 as atan2, array

def latlon_to_xyz(latlon):
    return array([sin(latlon[1])*cos(latlon[0]), -cos(latlon[0])*cos(latlon[1]), sin(latlon[0])])

def xyz_to_latlon(xyz):
    return array([arcsin(xyz[2]), atan2(xyz[0], -xyz[1])])