# -*- coding:utf-8 -*-

"""
    @File : __init__.py
    @Time : 2020/7/10 13:59
    @Author : sxwang
"""
from __future__ import absolute_import
from .unet import Residual_Unet
from .densenet import Densenet

__factory = {
    'unet': Residual_Unet,
    'densenet' : Densenet,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown models: {}".format(name))
    return __factory[name](*args, **kwargs)
