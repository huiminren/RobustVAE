#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:39:51 2019

@author: huiminren
"""


def shrink(epsilon,X_in):
    """
    @Original Author: Prof. Randy, Chong Zhou
    @Modified by: Huimin Ren

    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the tensor to shrink on

    Returns:
        The shrunk vector
    """
    x = X_in.clone()
    t1 = x > epsilon
    t2 = x < epsilon
    t3 = x > -epsilon
    t4 = x < -epsilon
    x[t2 & t3] = 0
    x[t1] = x[t1] - epsilon
    x[t4] = x[t4] + epsilon
    return x