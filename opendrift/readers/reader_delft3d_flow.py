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

from opendrift.readers.basereader import BaseReader, StructuredReader
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
    # Standard variable mapping for get_variables. Salinity and temperature
    # are grouped into dataset variable `R1`. They need to be expanded.
    standard_variable_mapping = {
        'time'                              : 'time',
        'longitude'                         : 'XZ',
        'latitude'                          : 'YZ',
        'land_binary_mask'                  : 'KCS',
        'sea_floor_depth_below_sea_level'   : 'DPS0',
        'sea_surface_height'                : 'S1',
        'x_sea_water_velocity'              : 'U1',
        'y_sea_water_velocity'              : 'V1',
        'upward_sea_water_velocity'         : 'WPHY',
        'water_density'                     : 'RHO',
        'sea_water_temperature'             : 'temp',
        'sea_water_salinity'                : 'salt',
    }

    zlevels = np.concatenate([
        np.arange(-10, -1,0.5),
        np.arange(-1, 1, 0.1),
        np.arange(1, 10, 0.5),
        np.arange(10, 50, 1),
        np.arange(50, 100, 5),
        np.arange(100, 500, 10),
        np.arange(500, 2500, 100),
        ],axis=0)

    def __init__(self,
        filename=None,
        name=None,
        proj4=None,
        custom_name_mapping={},
        zlevels=None,
        ensemble_member=None,
    ):
        self.time = None
        self.lon = None
        self.lat = None
        self.dimensions = {}
        self.proj4 = proj4
        if self.proj4 is not None:
            self._parse_proj4()
        if zlevels:
            self.zlevels = zlevels
        self._open_datastream(filename=filename,
                              name=name,
                              ensemble_member=ensemble_member)
        self.standard_variable_mapping.update(custom_name_mapping)
        self._get_independent_vars()
        self.variables = list(self.standard_variable_mapping)
        super().__init__()

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
        # Convert times to an array of datetime objects
        self.Dataset = xr.decode_cf(self.Dataset)

    def _parse_proj4(self):
        raise NotImplementedError(f"Unprojected coordinates not implemented"
            f" for {type(self)} Reader class")

    @staticmethod
    def _fill_masked_coords(x, y):
        """Fill in a value in the masked coordinates.

        Arguments
        ---------
        x: array_like
            With coordinates along the x-axis
        y: array_like
            With coordinates along the y-axis

        Returns
        -------
        tuple:
            With coordinate arrays filled with a value for the masked nodes.
        """
        # Opposite axis
        opp_axis = [1, 0]
        # Translation array for tiling
        tarray = np.array([
            [0, 1],
            [1, 0]
            ])
        for i, coord in enumerate([x, y]):
            # Place the anchor point 10% of the domain span to the left and
            # bottom of the domain
            unmasked = np.logical_not(coord.mask)
            anchor = coord.data[unmasked].min() \
                - 0.1 * (coord.data[unmasked].max() \
                - coord.data[unmasked].min())
            # Find the mean dx
            dx = np.abs(np.mean(np.diff(coord, axis=i)))
            # Make a linear incremented outer domain away from the anchor point
            outer = np.linspace(
                anchor,
                anchor - dx * (coord.shape[i] - 1),
                coord.shape[i]
            )
            # Tile to get the same shape as the coordinates
            outer = np.expand_dims(outer, axis=opp_axis[i])
            tile_shape = coord.shape * tarray[i,:] + np.flip(tarray[i, :])
            outer = np.tile(outer, tile_shape)
            # Map the masked with the exact number of cells from the scale
            coord.data[coord.mask] = outer[coord.mask]
        return x, y

    def _get_independent_vars(self):
        """Parses variable mapping and fetches time, x and y into object
        variables.
        """
        indvarnames = {'time': 'time', 'x': 'longitude', 'y': 'latitude'}
        mask_name = self.standard_variable_mapping['land_binary_mask']
        mask = self.Dataset[mask_name].data
        # Find valid min and max indices
        mask = np.logical_not(mask) # Originally False for masked values.
        varname = self.standard_variable_mapping['time']
        self.times = self.Dataset[varname].data.astype('M8[ms]') \
            .astype('O').tolist()
        varname = self.standard_variable_mapping['longitude']
        self.lon = np.ma.masked_where(mask, self.Dataset[varname].data)
        varname = self.standard_variable_mapping['latitude']
        self.lat = np.ma.masked_where(mask, self.Dataset[varname].data)
        self.lon, self.lat = self._fill_masked_coords(self.lon, self.lat)
        for key, val in indvarnames.items():
            varname = self.standard_variable_mapping[val]
            self.dimensions[key] = self.Dataset[varname].dims
        self.start_time = self.times[0]
        self.end_time = self.times[-1]
        if len(self.times) > 1:
            self.time_step = self.times[1] - self.times[0]
        else:
            self.time_step = None

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
        variables = {}
        start_time = datetime.now()
        nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = \
            self.nearest_time(time)
        requested_variables, time, x, y, z, outside = self.check_arguments(
            requested_variables, time, x, y, z)
        # For each variable
        print('time', time)
        print('nearest time, indxTime', nearestTime, indxTime)
        print('requested variables', requested_variables)
        print('x, y, z', x, y, z)
        for variable in requested_variables:
            # Map variable
            varname = self.standard_variable_mapping[variable]
            
            # Find nearest x, y, z
            # Destagger if needed
            # extract those profiles for these times
            # Do the vertical transformation for those profiles at this times
            # Interpolate each profile to the nearest z
        return variables
