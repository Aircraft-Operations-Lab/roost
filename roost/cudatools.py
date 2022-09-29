#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycuda.driver import *
import pycuda.gpuarray as gpuarray
import numpy as np


def format_4d_array_to_3d_f4(arr_i):
    s = arr_i.shape  # .swapaxes(0, 3, 2, 1)
    return arr_i.astype(np.float32).swapaxes(1, 3) \
                                   .reshape(s[0], s[3], -1, order='C')


def flip_weather_array_for_texture(arr_i):
    return np.ascontiguousarray(arr_i.swapaxes(1, 3), dtype=np.float32)


def np_to_array_3d_f4(nparray, allowSurfaceBind=False):
    dimension = len(nparray.shape)
    stride = 1
    d, h, w = nparray.shape
    descrArr = ArrayDescriptor3D()
    descrArr.width = w
    descrArr.height = h
    descrArr.depth = d
    
    descrArr.format = array_format.FLOAT #SIGNED_INT32 # Reading data as int4 (re=(hi,lo),im=(hi,lo)) structure
    descrArr.num_channels = 4

    if allowSurfaceBind:
        if dimension == 2:  descrArr.flags |= array3d_flags.ARRAY3D_LAYERED
        descrArr.flags |= array3d_flags.SURFACE_LDST

    cudaArray = Array(descrArr)
    copy3D = Memcpy3D()
    copy3D.set_src_host(nparray)
    copy3D.set_dst_array(cudaArray)
    copy3D.width_in_bytes = copy3D.src_pitch = nparray.strides[stride]
    copy3D.src_height = copy3D.height = h
    copy3D.depth = d
    copy3D()
    return cudaArray


def np_to_array_4d_f4(nparray, allowSurfaceBind=False):
    dimension = len(nparray.shape)
    stride = 1
    d, h, w = nparray.shape[:3]
    descrArr = ArrayDescriptor3D()
    descrArr.width = w
    descrArr.height = h
    descrArr.depth = d
    
    descrArr.format = array_format.SIGNED_INT32 # Reading data as int4 (re=(hi,lo),im=(hi,lo)) structure
    descrArr.num_channels = 4

    if allowSurfaceBind:
        if dimension==2:  descrArr.flags |= array3d_flags.ARRAY3D_LAYERED
        descrArr.flags |= array3d_flags.SURFACE_LDST

    cudaArray = Array(descrArr)
    copy3D = Memcpy3D()
    copy3D.set_src_host(nparray)
    copy3D.set_dst_array(cudaArray)
    copy3D.width_in_bytes = copy3D.src_pitch = nparray.strides[stride]
    copy3D.src_height = copy3D.height = h
    copy3D.depth = d
    copy3D()
    return cudaArray


def a2gpu(a, dtype=np.float32):
    return gpuarray.to_gpu(a.astype(dtype))


def get_next_power_of_two(n):
    if n:
        return 1 << (n-1).bit_length()
    else:
        return 1
