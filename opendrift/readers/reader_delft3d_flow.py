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
    A reader for Delft3D-Flow netCDF Output files. It can take a single file, a
    file pattern, a URL or an xarray Dataset.

    Args:
        :param filename: A single netCDF file, a pattern of files, or a
                         xr.Dataset. The netCDF file can also be an URL to an
                         OPeNDAP server.
        :type filename: string, xr.Dataset (required).

        :param name: Name of reader
        :type name: string, optional

        :param proj4: PROJ.4 string describing projection of data.
        :type proj4: string, optional


    Example:

    .. code::

       from opendrift.readers.reader_delft3d_flow_ import Reader
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
    'u_rho': 'eastward_sea_water_velocity',
    'v_rho': 'northward_sea_water_velocity',
    'w_rho': 'upward_sea_water_velocity',
    'temp': 'sea_water_temperature',
    'salt': 'sea_water_salinity',
    'tke': 'turbulent_kinetic_energy',
    }

    def __init__(self,
        filename=None,
        name=None,
        proj4=None,
        custom_name_mapping={},
        zlevels=None,
        ensemble_member=None,
    ):
        if self.proj4 is None:
            self.proj4 = '+proj=latlong'
            self.projected = False
        else:
            self._parse_proj4()
        self._open_datastream(filename=filename,
                              name=name,
                              ensemble_member=ensemble_member)
        self._map_d3d_variable_names(custom_name_mapping)

    def _open_datastream(self,
        filename=None,
        name=None,
        ensemble_member=None,
    ):
        assert filename is not None, 'Need filename as argument to constructor'
        namestr = None
        filestr = str(filename)
        if isinstance(filename, xr.Dataset):
            self.Dataset = filename
            if hasattr(self.Dataset, 'name'):
                namestr = self.Dataset.name
        else:
            try:
                # Open file, check that everything is ok
                logger.info('Opening dataset: ' + filestr)
                if ('*' in filestr) or ('?' in filestr) or ('[' in filestr):
                    logger.info('Opening files with MFDataset')
                    self.Dataset = xr.open_mfdataset(filename,
                                                     data_vars='minimal',
                                                     coords='minimal',
                                                     chunks={'time': 1},
                                                     decode_times=False)
                elif ensemble_member is not None:
                    self.Dataset = (xr.open_dataset(filename,
                                                   decode_times=False)
                                   .isel(ensemble_member=ensemble_member))
                else:
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

    def _parse_proj4(self):
        pass

    def _map_d3d_variable_names(self, custom_mapping):
        self.standard_variable_mapping.update(custom_mapping)
        # Map ungrouped variables

        # Map grouped variables

    @staticmethod
    def _get_variable_coordinates(ds, var):
        """Finds the `ds` xarray dataset coordinate names for the variable
        `var`.

        Arguments
        ---------
        ds: x-array dataset
            An x-array dataset containing the variable `var`.
        var: str
             name of the variable to get the coordinates for.

        Returns
        -------
        list:
            With the coordinates of each of the variable's axis in the same
            order as `ds[var].dims`.
        """
        coord_switch = {'M': 'x', 'N': 'y'}
        coordsout = list(range(len(ds[var].dims))) # Needs to be in the same
                                                   # order as dims.
        for i, dim in enumerate(ds[var].dims):
            # List the coordinates that have this dimension
            dimcoords = list(filter(lambda x: dim in ds[x].dims,
                                   ds.coords.keys()))
            try:
                axis = coord_switch[dim[0]] + '-coordinate'
            except KeyError:
                # Any other dimension has the same name as the coordinate
                coordsout[i] = dim
            else:
                # Find the coordinate of the correct axis
                coordsout[i] = list(filter(lambda x: axis in
                    ds.coords[x].long_name.lower(),
                    dimcoords))[0]
        return coordsout

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

        # Destagger all variables from their grids to rho grid
        pass
