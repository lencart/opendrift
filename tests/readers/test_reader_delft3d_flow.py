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
        d3d_fn = o.test_data_folder() + 'delft3d_flow/trim-f34_nc.nc'
        myreader = reader_delft3d_flow.Reader(filename=d3d_fn)

if __name__ == '__main__':
    unittest.main()

