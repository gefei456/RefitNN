#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         11:41
# @Author:      WGF
# @File:        other_func.py
# @Description:

import numpy as np
def pos_screen2std(Pos, screen_W, screen_H):
    Pos_new = np.zeros(2,)
    Pos_new[0] = Pos[0] - screen_W / 2
    Pos_new[1] = -Pos[1] + screen_H / 2
    return Pos_new

def trial_cell_to_dict(expData):
    multi_cell_array = expData["trialData"]
    dictList = []
    for row in multi_cell_array:
        for element in row:
            dict = {}
            for name in element.dtype.names:
                if element[name][0][0].size > 0:
                    dict[name] = element[name][0][0][0]
                else:
                    dict[name] = element[name][0][0]
            dictList.append(dict)
    return dictList