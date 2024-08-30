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

from bisect import bisect_left, bisect_right
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

import numpy as np
from netCDF4 import num2date
import xarray as xr

from opendrift.readers.basereader import BaseReader, vector_pairs_xy, StructuredReader
from opendrift.readers.roppy import depth


class Reader(BaseReader, StructuredReader):
    """
    A reader for ROMS Output files. It can take a single file, a file pattern,
    a URL or an xarray Dataset.

    Args:
        :param filename: A single netCDF file, a pattern of files, or a
                         xr.Dataset. The netCDF file can also be an URL to an
                         OPeNDAP server.
        :type filename: string, xr.Dataset (required).

        :param name: Name of reader
        :type name: string, optional

        :param save_interpolator: Whether or not to save the interpolator that
                                  goes from lon/lat to x/y (calculated in structured.py)
        :type save_interpolator: bool

        :param interpolator_filename: If save_interpolator is True, user can
                                      input this string to control where
                                      interpolator is saved.
        :type interpolator_filename: Path, str, optional

    Example:

    .. code::

       from opendrift.readers.reader_ROMS_native import Reader
       r = Reader("roms.nc")

    Several files can be specified by using a pattern:

    .. code::

       from opendrift.readers.reader_ROMS_native import Reader
       r = Reader("*.nc")

    An OPeNDAP URL can be used:

    .. code::

       from opendrift.readers.reader_ROMS_native import Reader
       r = Reader('https://thredds.met.no/thredds/dodsC/mepslatest/meps_lagged_6_h_latest_2_5km_latest.nc')

    A xr.Dataset can be used:

    .. code::

        from opendrift.readers.reader_ROMS_native import Reader
        ds = xr.open_dataset(filename, decode_times=False)
        r = Reader(ds)
    """

    standard_variable_mapping = {
    'mask_rho': 'land_binary_mask',
    'mask_psi': 'land_binary_mask',
    'h': 'sea_floor_depth_below_sea_level',
    'zeta': 'sea_surface_height',
    'u': 'x_sea_water_velocity',
    'v': 'y_sea_water_velocity',
    'u_eastward': 'eastward_sea_water_velocity',
    'v_northward': 'northward_sea_water_velocity',
    'w': 'upward_sea_water_velocity',
    'temp': 'sea_water_temperature',
    'salt': 'sea_water_salinity',
    'uice': 'sea_ice_x_velocity',
    'vice': 'sea_ice_y_velocity',
    'aice': 'sea_ice_area_fraction',
    'hice': 'sea_ice_thickness',
    'gls': 'turbulent_generic_length_scale',
    'tke': 'turbulent_kinetic_energy',
    'AKs': 'ocean_vertical_diffusivity',
    'sustr': 'surface_downward_x_stress',
    'svstr': 'surface_downward_y_stress',
    'tair': 'air_temperature',
    'wspd': 'wind_speed',
    'uwnd': 'x_wind',
    'vwnd': 'y_wind',
    'uwind': 'x_wind',
    'vwind': 'y_wind',
    'Uwind': 'x_wind',
    'Vwind': 'y_wind',
    }
    def __init__(self,
        filename=None,
        name=None,
        proj4=None,
        custom_name_mapping={},
        zlevels=None,
        ensemble_member=None,
    ):
        self._open_datastream(filename=filename,
                              name=name,
                              ensemble_member=ensemble_member)

    def _open_datastream(self,
        filename=None,
        name=None,
        ensemble_member=None,
    ):
        assert filename is not None, 'Need filename as argument to constructor'
        namestr = None
        filestr = str(filename)
        if isinstance(filename, xr.Dataset):
            print("##########OPENING XARRAY#######")
            self.Dataset = filename
            if hasattr(self.Dataset, 'name'):
                namestr = self.Dataset.name
        else:
            try:
                # Open file, check that everything is ok
                logger.info('Opening dataset: ' + filestr)
                if ('*' in filestr) or ('?' in filestr) or ('[' in filestr):
                    print("##########OPENING FILE 1#######")
                    logger.info('Opening files with MFDataset')
                    self.Dataset = xr.open_mfdataset(filename,
                                                     data_vars='minimal',
                                                     coords='minimal',
                                                     chunks={'time': 1},
                                                     decode_times=False)
                elif ensemble_member is not None:
                    print("##########OPENING FILE 2#######")
                    self.Dataset = (xr.open_dataset(filename,
                                                   decode_times=False)
                                   .isel(ensemble_member=ensemble_member))
                else:
                    print("##########OPENING FILE 3#######")
                    self.Dataset = xr.open_dataset(filename,
                                                   decode_times=False)
            except Exception as e:
                raise ValueError(e)
        if name is not None:
            self.name = name
        elif namestr is not None:
            self.name = namestr
        else:
            self.name = filestr

    def get_variables(self,
        requested_variables,
        time=None,
        x=None,
        y=None,
        z=None,
        testing=False
    ):
        start_time = datetime.now()
        requested_variables, time, x, y, z, outside = self.check_arguments(
            requested_variables, time, x, y, z)
        pass
