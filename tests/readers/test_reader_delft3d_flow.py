#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

import os
from datetime import datetime
import unittest


from opendrift.readers import reader_delft3d_flow
#from opendrift.readers import reader_global_landmask
from opendrift.models.oceandrift import OceanDrift

class TestDelft3D(unittest.TestCase):

    def test_open_datastream(self):
        o = OceanDrift(loglevel=30)
        d3d_fn = o.test_data_folder() + 'delft3d_flow/trim-f34_wgs84.nc'
        myreader = reader_delft3d_flow.Reader(filename=d3d_fn)

    def test_get_variable_coordinates(self):
        o = OceanDrift(loglevel=30)
        d3d_fn = o.test_data_folder() + 'delft3d_flow/trim-f34_wgs84.nc'
        myreader = reader_delft3d_flow.Reader(filename=d3d_fn)
        h_coords = myreader._get_variable_coordinates(myreader.Dataset, 'S1')
        u_coords = myreader._get_variable_coordinates(myreader.Dataset, 'U1')
        v_coords = myreader._get_variable_coordinates(myreader.Dataset, 'V1')
        w_coords = myreader._get_variable_coordinates(myreader.Dataset, 'W')
        assert h_coords == ['time', 'XZ', 'YZ'], h_coords
        assert u_coords == ['time', 'KMAXOUT_RESTR', 'XCOR', 'YZ'], u_coords
        assert v_coords == ['time', 'KMAXOUT_RESTR', 'XZ', 'YCOR'], v_coords
        assert w_coords == ['time', 'KMAXOUT', 'XZ', 'YZ'], w_coords

    def test_add_reader(self):
        o = OceanDrift(loglevel=0)
        d3d_fn = o.test_data_folder() + 'delft3d_flow/trim-f34_wgs84.nc'
        myreader = reader_delft3d_flow.Reader(filename=d3d_fn)
        o.add_reader(myreader)
        

if __name__ == '__main__':
    unittest.main()

