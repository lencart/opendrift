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
import numpy as np

from opendrift.readers import reader_delft3d_flow
#from opendrift.readers import reader_global_landmask
from opendrift.models.oceandrift import OceanDrift

class TestDelft3D(unittest.TestCase):

    def test_open_datastream(self):
        o = OceanDrift(loglevel=30)
        d3d_fn = o.test_data_folder() + 'delft3d_flow/trim-f34_wgs84.nc'
        myreader = reader_delft3d_flow.Reader(filename=d3d_fn)
        print(myreader)

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

    def test_seed_run(self):
        o = OceanDrift(loglevel=30)
        d3d_fn = o.test_data_folder() + 'delft3d_flow/trim-f34_wgs84.nc'
        myreader = reader_delft3d_flow.Reader(filename=d3d_fn)
        o.add_reader(myreader)
        print("start_time", myreader.start_time)
        o.seed_elements(lat=53.52, lon=6.0, radius=0, number=10,
                z=np.linspace(0, -1, 10), time=myreader.start_time)
        o.run(time_step=15*60, steps=10)
        return o, myreader

    def test_projected(self):
        o = OceanDrift(loglevel=30)
        d3d_fn = o.test_data_folder() + 'delft3d_flow/trim-f34_wgs84.nc'
        expected = NotImplementedError
        try:
            myreader = reader_delft3d_flow.Reader(filename=d3d_fn, proj4='astr')
        except Exception as err:
            assert isinstance(err, expected), (f"This raises a "
                f"{type(err)} error instead of {expected}")
            print(f"Got {err}")

    def test_get_coordinates(self, ts, xs, ys, zs):
        o = OceanDrift(loglevel=0)
        d3d_fn = o.test_data_folder() + 'delft3d_flow/trim-f34_wgs84.nc'
        myreader = reader_delft3d_flow.Reader(filename=d3d_fn)
        return myreader, myreader._get_depth_coords(
            ts,
            xs,
            ys,
            zs,
        )

    def test_multi_zslice(self):
        o = OceanDrift(loglevel=0)
        d3d_fn = o.test_data_folder() + 'delft3d_flow/trim-f34_wgs84.nc'
        r = reader_delft3d_flow.Reader(filename=d3d_fn)
        r.buffer = 0
        xs = [5.6, 18.2]
        ys = [2.2, 9.8]
        zs = [0., -.999999]
        t = 0
        indx, indy = r._get_xy(xs, ys)
        xx, yy = np.meshgrid(indx, indy)
        zlv, zsg = r._get_depth_coords(t, xx.flatten(), yy.flatten(), zs)
        u = r.Dataset['U1'].data
        # Ask again for the indentity interpolation of z at sigma against
        # z at sigma
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                # Make the zlevels the exact same as the sigma layers at the
                # same horizontal indices of the cube we are requesting.
                # This is need to guaranty that the calculated target zlevels
                # align perfectly with the z at sigma so that we can validate
                # the indentity operation.
                r.zlevels = zsg[:, i*j]
                u_int = r._interpolate_profile(
                    'U1',
                    xx[i, j],
                    yy[i, j],
                    zsg[:,i*j],
                    zsg[(0, -1), i*j],
                    itime=25,
                )
                assert np.allclose(u[25, :, yy[i, j], xx[i, j]], u_int.T), \
                    f"Breaks in ({i}, {j})"
        return r, xx, yy, zlv, zsg, u, u_int

if __name__ == '__main__':
    unittest.main()

