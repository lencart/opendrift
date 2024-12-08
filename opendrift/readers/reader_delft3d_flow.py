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

import copy
from bisect import bisect_left, bisect_right
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

import numpy as np
from netCDF4 import num2date
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
import pyproj

from opendrift.readers.basereader import BaseReader, StructuredReader
from opendrift.readers.basereader.fakeproj import fakeproj
from opendrift.readers.roppy import depth as drp

def nearest(array, value, n=1):
    """Find the indices of the `n` elements `array` nearest to value(s).

    Parameters
    ----------
    array : array-like
        Array with the values to compare against `value`.
    value : float, array-like
        Value(s) evaluate the distance to `array`. This can either be a single
        value or a vector (1d array) with multiple values.
    n : int (optional, default = 1)
        Number of indices to return by order of proximity to `value`.
    
    Returns
    -------
    array-like :
         Array with `n * value.shape[0]` number of indices of `array` by order
         of proximity to `value`.
    """
    value = np.atleast_1d(np.squeeze(value))
    array = np.atleast_2d(array)
    assert value.ndim == 1, (f"value has to be a single dimension vector"
        f" but has dimensions {value.shape}")
    ndim = array.ndim
    shape = array.shape
    # Extend array to allow broacasting subtraction with vector value.
    array = array[..., None]
    # Extend vector value with the extra array dims.
    value = np.asarray(value)[ndim * (None,) + (...,)]
    # Calculate element-wise distance and reshape into flat array shapes
    # stacked for each of the elements of the value vector representing
    # the distances of each array element to the value vector element
    # at the same position.
    idx = np.abs(array - value).reshape(np.multiply(*shape), -1)
    # Sort each of the distances to get the index of the closest array element.
    idx = idx.argsort(axis=0)
    return np.squeeze(idx[:n,:].T)

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
        'sigma'                             : 'SIG_LYR',
        'land_binary_mask'                  : 'KCS',
        'sea_floor_depth_below_sea_level'   : 'DPS0',
        'sea_surface_height'                : 'S1',
        'x_sea_water_velocity'              : 'U1',
        'y_sea_water_velocity'              : 'V1',
        'sea_water_speed'                   : 'U1', #dummy for _get_data_method
        'upward_sea_water_velocity'         : 'WPHY',
        'water_density'                     : 'RHO',
        'sea_water_temperature'             : 'R1',
        'sea_water_salinity'                : 'R1',
    }

    _r1_variables = {
        'sea_water_temperature'             : 1,
        'sea_water_salinity'                : 0,
    }

    _to_destagger = {
        'x_sea_water_velocity'              : 'U1',
        'y_sea_water_velocity'              : 'V1',
    }

    _mask_names = {'KCS', 'KFU', 'KFV'}

    zlevels = -1 * np.concatenate([
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
        # unrotated velocity cache
        self._uv_cache = {}
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
        self._overload_xy_grid()

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

    def __repr__(self):
        """Overloads same method in `variables`"""
        outStr = '===========================\n'
        outStr += 'Reader: ' + self.name + '\n'
        outStr += 'Projection: \n  ' + self.proj4 + '\n'
        outStr += 'Coverage: [%s]\n' % self._coverage_unit_()
        shape = self.shape
        if shape is None:
            outStr += '  xmin: %f   xmax: %f\n' % (self.xmin, self.xmax)
            outStr += '  ymin: %f   ymax: %f\n' % (self.ymin, self.ymax)
        else:
            outStr += '  xmin: %f   xmax: %f   step: %g   numx: %i\n' % \
                (self.xmin, self.xmax, self.delta_x or 0, shape[0])
            outStr += '  ymin: %f   ymax: %f   step: %g   numy: %i\n' % \
                (self.ymin, self.ymax, self.delta_y or 0, shape[1])
        corners =[
            (np.nanmin(self.lon), np.nanmin(self.lat)),
            (np.nanmax(self.lon), np.nanmin(self.lat)),
            (np.nanmin(self.lon), np.nanmax(self.lat)),
            (np.nanmax(self.lon), np.nanmax(self.lat)),
        ]
        outStr += '  Corners (lon, lat):\n'
        outStr += '    (%6.2f, %6.2f)  (%6.2f, %6.2f)\n' % \
            (corners[2][0],
             corners[2][1],
             corners[3][0],
             corners[3][1])
        outStr += '    (%6.2f, %6.2f)  (%6.2f, %6.2f)\n' % \
            (corners[0][0],
             corners[0][1],
             corners[1][0],
             corners[1][1])
        if hasattr(self, 'z'):
            with np.printoptions(suppress=True) as opts:
                outStr += 'Vertical levels [m]: \n  ' + str(self.z) + '\n'
        elif hasattr(self, 'sigma'):
            with np.printoptions(suppress=True, threshold=20) as opts:
                    outStr += (
                        f'Vertical levels [sigma]: \n'
                        f' {self.sigma}\n')
                    outStr += (
                        f'Interpolated into zeta [m]:\n'
                        f' {self.zlevels}\n')
        else:
            outStr += 'Vertical levels [m]: \n  Not specified\n'
        outStr += 'Available time range:\n'
        outStr += '  start: ' + str(self.start_time) + \
                  '   end: ' + str(self.end_time) + \
                  '   step: ' + str(self.time_step) + '\n'
        if self.start_time is not None and self.time_step is not None:
            outStr += '    %i times (%i missing)\n' % (
                self.expected_time_steps, self.missing_time_steps)
        if hasattr(self, 'realizations') and self.realizations is not None:
            outStr += 'Variables (%i ensemble members):\n' % len(
                self.realizations)
        else:
            outStr += 'Variables:\n'
        for variable in self.variables:
            if variable in self.derived_variables:
                outStr += '  ' + variable + ' - derived from ' + \
                    str(self.derived_variables[variable]) + '\n'
            else:
                outStr += '  ' + variable + '\n'
        outStr += '===========================\n'
        outStr += self.performance()

        return outStr

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
        outers = []
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
            outers.append(outer)
        return x.data, y.data, outers

    def _get_independent_vars(self):
        """Parses variable mapping and fetches time, x and y into object
        variables.
        """
        indvarnames = {
            'time': 'time',
            'x': 'longitude',
            'y': 'latitude',
            'z': 'sigma',
        }
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
        self._lon_mask = self.lon.mask.copy()
        self._lat_mask = self.lat.mask.copy()
        self.lon, self.lat, _ = self._fill_masked_coords(self.lon, self.lat)
        try:
            varname = self.standard_variable_mapping['sigma']
            self.sigma = self.Dataset[varname].data
        except KeyError:
            raise NotImplementedError(('Delft3d-Flow reader only implemented'
                ' for 3d vertical sigma coordinates')
                )
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
    def _get_var_coords(ds, var):
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

    def _overload_xy_grid(self):
        if isinstance(self.proj, fakeproj):
            logger.warning(
                (
                "No proj string or projection could be derived, using"
                " 'fakeproj'. This assumes that the variables are structured"
                " and gridded approximately equidistantly on the surface"
                " (i.e. in meters). This must be guaranteed by the user. You"
                " can get rid of this warning by supplying a valid projection"
                " to the reader."
                )
            )
            # Making interpolator (lon, lat) -> x
            # save to speed up next time
            if self.save_interpolator and self.interpolator_filename is not None:
                interpolator_filename = Path(self.interpolator_filename).with_suffix('.pickle')
            else:
                interpolator_filename = f'{self.name}_interpolators.pickle'
            if self.save_interpolator and Path(interpolator_filename).is_file():
                logger.info((
                        "Loading previously saved interpolator for lon,lat"
                        " to x,y conversion."
                    )
                )
                with open(interpolator_filename, 'rb') as file_handle:
                    interp_dict = pickle.load(file_handle)
                    spl_x = interp_dict["spl_x"]
                    spl_y = interp_dict["spl_y"]
            else:
                logger.info((
                        "Making interpolator for lon,lat to x,y"
                        " conversion..."
                    )
                )
                block_x, block_y = np.mgrid[self.xmin:self.xmax + 1,
                                            self.ymin:self.ymax + 1]
                mask = np.logical_not(self._lon_mask)
                block_x, block_y = block_x.T, block_y.T
                spl_x = LinearNDInterpolator(
                    (self.lon[mask].ravel(), self.lat[mask].ravel()),
                    block_x[mask].ravel(),
                    fill_value=np.nan)
                # Reusing x-interpolator (deepcopy) with data for y
                spl_y = copy.deepcopy(spl_x)
                spl_y.values[:, 0] = block_y[mask].ravel()
                # Call interpolator to avoid threading-problem:
                # https://github.com/scipy/scipy/issues/8856
                spl_x((0, 0)), spl_y((0, 0))
                if self.save_interpolator:
                    logger.info((
                        "Saving interpolator for lon,lat to x,y"
                        " conversion."
                        )
                    )
                    interp_dict = {"spl_x": spl_x, "spl_y": spl_y}
                    with open(interpolator_filename, 'wb') as f:
                        pickle.dump(interp_dict, f)
            self.spl_x = spl_x
            self.spl_y = spl_y

    def xy2lonlat(self, x, y):
        """Overloads structured reader function.
        """
        if self.projected:
            return super().xy2lonlat(x, y)
        else:
            mask = self._lon_mask
            np.seterr(invalid='ignore')  # Disable warnings for nan-values
            y = np.atleast_1d(y).astype('float64')
            x = np.atleast_1d(x).astype('float64')
            # NB: mask coordinates outside domain
            x[x < self.xmin] = np.nan
            x[x > self.xmax] = np.nan
            y[y < self.ymin] = np.nan
            y[y < self.ymin] = np.nan
            lon_in = self.lon.copy()
            lat_in = self.lat.copy()
            lon = map_coordinates(lon_in, [y, x],
                                  order=1,
                                  cval=np.nan,
                                  mode='nearest')
            lat = map_coordinates(lat_in, [y, x],
                                  order=1,
                                  cval=np.nan,
                                  mode='nearest')
            return (lon, lat)

    def pixel_size(self):
        """Overloads the same method from structured to return the minimum
        pixel size"""
        if self.projected:
            raise NotImplementedError((
                    "Delft3D-Flow reader not implemented for"
                    " projected coordinates"
                )
            )
        else:
            mask = self._lon_mask
            lons = self.lon.copy()
            lats = self.lat.copy()
            # Fill in the mask with very large numbers so that the distance
            # between a valid node and a masked one is huge and that node
            # is not selected as the smallest distance.
            lons[mask] = 1e6
            lats[mask] = 1e6
            geod = pyproj.Geod(ellps='WGS84')  # Define an ellipsoid
            dist_x = geod.inv(lons[:,:-1], lats[:,:-1], lons[:,1:], lats[:,1:],
                radians=False)[2]
            dist_y = geod.inv(lons[:-1,:], lats[:-1,:], lons[1:,:], lats[1:,:],
                radians=False)[2]
            mindists = [np.nanmin(dist_x), np.nanmin(dist_y)]
            # Take the minimum distance and divide by the dimension of that
            # fakeproj axis.
            pixelsize = np.mean(mindists) #/ self.shape[np.argmin(mindists)]
            return pixelsize

    def modulate_longitude(self, lons):
        """
        Modulate the input longitude to the domain supported by the reader.

        Overloads parent method class to bypass checking for sign.
        """
        # Delft 3d longitudes are positive east from Greenwich.
        # This method overloads the parent that requests longitudes for xy
        # fakeproje xtents that can lead into np.nans outside mask
        return  np.mod(lons + 180, 360) - 180

    def _get_depth_coords(self, cube_slice, z_targets):
        """Finds the nearest zlevels to the target zlevels and calculates the
        depth for each of the (t, x, y) tuples.

        Arguments
        ---------
        t: int
            Time index when to look for the depth coordinates
        xs, ys: array_like
            With the indices for target grid cells
        z_targets: numpy.array, None
            Requested approximate depths to avaluate the variables. If it is
            None, it triggers interpolation to the minimum z-level (deepest) of
            the surface sigma layer.

        Returns
        -------
        zlevels: array_like
            With the resulting nearest z-level to each of the `z_targets`.
        z_at_sigma: array_like
            With depths at sigma coordinates for each of the (t, x, y) tuple.
        """
        # Dynamic depth at (t, x, y)
        eta = self._get_cube('sea_surface_height', cube_slice)[0]
        dep = self._get_cube('sea_floor_depth_below_sea_level', cube_slice[-2:])[0]
        zeta = dep + eta
        # z levels at sigma coordinates
        zs = self.sigma[:, None, None] * zeta
        # Get all the zlevels within the range of z_targets
        try:
            # Top level is the deepest z-level below deepest top-level z-sigma
            z_max = np.min([z_targets.max(), np.nanmin(zs[0, ...]).min()])
            # lowest level is the shallowest z-level above deepest bottom-level
            # z-sigma
            z_min = np.max([z_targets.min(), np.nanmin(zs[-1, ...]).min()])
            z_range = np.array([z_min, z_max])
            print('ZRANGE', z_range)
        except AttributeError:
            # When z is none, return just the 1st sigma level
            z_range =  np.atleast_1d(np.nanmin(zs[0, ...]))
            i_range = np.sort(np.searchsorted(-1 * self.zlevels, -1 * z_range))
            z_targets = self.zlevels[i_range]
        else:
            i_range = np.sort(np.searchsorted(-1 * self.zlevels, -1 * z_range))
            # Make it one less at the deepest end so that there are sigma there
            # to interpolate from.
            z_targets = self.zlevels[i_range[0]: i_range[-1]]
        return z_targets, zs.reshape(self.sigma.shape[0], -1)

    def _get_xy(self, x, y):
        """Calculates the target horizontal grid indices.
        """
        if hasattr(self, 'clipped'):
            clipped = self.clipped
        else: 
            clipped = 0
        indx = np.floor(np.abs(x-self.x[0])/self.delta_x-clipped).astype(int) + clipped
        indy = np.floor(np.abs(y-self.y[0])/self.delta_y-clipped).astype(int) + clipped
        buffer = self.buffer  # Adding buffer, to cover also future positions of elements
        indy = np.arange(np.max([0, indy.min() - buffer]),
                         np.min([indy.max() + buffer + 1, self.numy]))
        indx = np.arange(indx.min() - buffer, indx.max() + buffer + 1)
        if self.global_coverage() and indx.min() < 0 and indx.max() > 0 and indx.max() < self.numx:
            logger.debug('Requested data block is not continuous in file'+
                          ', must read two blocks and concatenate.')
            indx_left = indx[indx<0] + self.numx  # Shift to positive indices
            indx_right = indx[indx>=0]
            if indx_right.max() >= indx_left.min():  # Avoid overlap
                indx_right = np.arange(indx_right.min(), indx_left.min())
        else:
            indx = np.arange(np.max([0, indx.min()]),
                             np.min([indx.max() + 1, self.numx]))
        return indx, indy

    def _interpolate_profile(self,
        data,
        zs_at_sigma,
        z_targets,
        itime=None,
    ):
        """Interpolates `data`  1D profile or a 2D sequence of N profiles
        (shape == (M, N)), for one grid (x, y) index (1D) or a set of N grid
        (xs, ys) indices, from their depth at sigma coordinates (`z_at_zigma`)
        to target depths `z_targets` for a single time index `itime`.

        Arguments
        ---------
        data: numpy.ndarray
            With the values to interpolate arranged in (M,) for a single
            profile or (M, N) for a sequence of profiles.
        zs_at_sigma: array_like
            Depths at the sigma coordinate for each (itime, x, y), positive-up
            from reference level. Has shape (M, N) with N == 1 if `xs` and `ys`
            are integers.
        z_targets: int, array_like
            Depth or depths at which to interpolate `varname` values into.

        Returns
        -------
        profiles: numpy.array
            Interpolation result with `profiles.shape == (K, *xs.shape)` where
            `K` is the number of self.zlevels within the range of `z_targets`.
        """
        # Interpolate profiles from z at sigma to target zlevels.
        # ROMS plofile interpolator only works with positive domains and
        # counterdomains.
        profiles, _ = drp.multi_zslice(data, -1 * zs_at_sigma, -1 * z_targets)
        # Mask profiles when z levels are outside of the z at sigma range.
        #if xs.ndim > 1:
        try:
            # It will work if there are multiple profiles to calculate
            mask = np.squeeze(
                np.logical_or(
                    z_targets[:, None] < zs_at_sigma.min(axis=0)[None, :],
                    z_targets[:, None] > zs_at_sigma.max(axis=0)[None, :],
                )
            ).reshape(profiles.shape)
        except IndexError:
            # This will raise an error if there is only one profile
            mask = np.logical_or(
                z_targets < zs_at_sigma.min(),
                z_targets > zs_at_sigma.max(),
            ).reshape(profiles.shape)
        profiles[mask] = np.nan
        # Return profiles in the shape of the requested cube
        return profiles

    def _get_cube(self, varname, cube, testing=False):
        """Returns a cube of `varname` data. Unpacks data from grouped
        variables, masks it and de-staggers it if needed.

        Arguments
        ---------
        varname: str
            Name of the variable as in the keys of
            `self.standard_variable_mapping`.
        cube: tuple
            Of `slice` elements With the extent of the data subset, in the
            following order: (times, zs, ys, xs).

        Warning
        -------
        All elements of the tuple must be `slice` to preserve dimensionality.

        Returns
        -------
        data: numpy.ma.MaskedArray
            Data cube containing the `varname` values for the slices in
            `cube`, masked with its respective mask.
        """
        get_data_method = {
            'R1': self._get_r1_data,
            'U1': self._get_uv_data,
            'V1': self._get_uv_data,
        }
        print("##################################")
        print("####### IN CUBE ##################")
        print("##################################")
        print("Asked for", varname, "with slice", cube)
        d3d_varname = self.standard_variable_mapping[varname]
        try:
            # Try if there are special get data methods
            data, cube = get_data_method[d3d_varname](
                varname,
                cube,
                testing=testing
            )
        except KeyError:
            data = self.Dataset[d3d_varname].data[cube]
        # Slices depending on number of dimensions of cube
        dim_slices = {
            # 3D-H cubes (time, x, y) have static masks
            3: tuple(cube[-2:]),
            # 4D cubes (time, z, x, y) masks don't have z dimension
            4: tuple([cube[0], *cube[-2:]]),
        }
        # Slices for cube expansion
        exp_slices = {
            3: (data.shape[0], 1, 1),
            4: (1, data.shape[1], 1, 1)
        }
        # Get mask for this variable, reducing the vertical dimension
        # when present.
        try:
            mask_cube = dim_slices[len(cube)]
        except KeyError:
            # 2D  variable
            mask_cube = cube
        mask = self._get_mask(varname, mask_cube)
        # Expand the vertical dimension
        try:
            mask = np.tile(mask, exp_slices[len(cube)]).reshape(data.shape)
        except KeyError:
            pass
        data[mask] = np.nan
        return data, mask

    def _get_r1_data(self, varname, cube, **kwargs):
        # Go into `R1` and extract cube
        i_data = self._r1_variables[varname]
        # Assumes always a 4D cube for R1 variables
        r1_cube = (cube[0], i_data, *cube[2:])
        data = self.Dataset['R1'].data[r1_cube]
        cube = (cube[0], *cube[2:])
        return data, cube

    def _get_uv_data(self, varname, cube, alfas_name='ALFAS', testing=False):
        uv_dict = {}
        try:
            return self._uv_cache[varname], cube
        except KeyError:
            # Get grid angles
            alfas = self.Dataset[alfas_name].data[cube[-2:]]
            # Get components
            for var, d3dvar in self._to_destagger.items():
                # Destagger
                uv_dict[var] = self._destagger(d3dvar, cube, testing=testing)
            # Rotate
            U = uv_dict['x_sea_water_velocity'] * np.cos(alfas * np.pi / 180) \
                - uv_dict['y_sea_water_velocity'] * np.sin(alfas * np.pi / 180)
            V = uv_dict['x_sea_water_velocity'] * np.sin(alfas * np.pi / 180) \
                + uv_dict['y_sea_water_velocity'] * np.cos(alfas * np.pi / 180)
            if testing:
                # Override rotation
                U = uv_dict['x_sea_water_velocity']
                V = uv_dict['x_sea_water_velocity']
            self._uv_cache['x_sea_water_velocity'] = U
            self._uv_cache['y_sea_water_velocity'] = V
            self._uv_cache['sea_water_speed'] = np.sqrt(U**2 + V**2)
            # Call again to return data
            data, cube = self._get_uv_data(varname, cube)
            return data, cube

    def _destagger(self, d3d_varname, cube, testing=False):
        """
        Put variable into water level grid

        This 1st attempt only works for 4-D variables
        """
        data = self.Dataset[d3d_varname].data[cube]
        return data

    def _get_mask(self, varname, cube):
        """
        Returns the mask of `varname`, sliced by `cube`.

        varname: str
            Name of the variable as in the keys of
            `self.standard_variable_mapping`.
        cube: tuple
            With the extent of the mask subset.

        Returns
        -------
        numpy.ma.MaskedArray
           Mask of `varname` for the slices in `cube`, masked with its
           respective mask.
        """
        print('cube slice asked in mask', cube)
        try:
            varname = self.standard_variable_mapping[varname]
        except KeyError:
            # Except 'raw' d3d names for testing purposes
            pass

        # Get horizontal coordinates for varname
        try:
            coords = self._get_var_coords(self.Dataset, varname)[-2:]
        except KeyError:
            # Handle R1 grouped variables
            coords = self._get_var_coords(self.Dataset, 'R1')[-2:]
        # Find the mask that contains all of the coordinates
        for mask_name in self._mask_names:
            # Get the horizontal coordinates for the mask
            mcoords = self._get_var_coords(self.Dataset, mask_name)[-2:]
            if set(coords).intersection(set(mcoords)) == set(mcoords):
                try:
                    # See if the cube fits
                    return self.Dataset[mask_name].data[cube] != 1
                except IndexError:
                    # If not, use just the horizontal coords but return the
                    # mask with the same shape as the cube. This will work
                    # for 3D or 4D variables with masks that do not vary with
                    # time: e.g WPHY.
                    out = self.Dataset[mask_name].data[cube[-2:]] != 1
                    newshape = (*(len(cube) * [1]), *out.shape[-2:])
                    return out.reshape(newshape)

        # Return empty array if mask not found
        return []


    def get_variables(self,
        requested_variables,
        time=None,
        x=None,
        y=None,
        z=None,
        testing=False
    ):
        """Returns the variables`requested_variables` with cubes spaning the
        hyper bounding box of `x`, `y`, `z` and t as the closest time index to
        `time`.

        Arguments
        ---------
        requested_variables: list
            Elements are strings belonging to
            `self.standard_variable_mapping.keys()`.
        time: datetime
            Date to approximate reader result.
        x: array_like
            Elements are `float` with projected x-axis coordinate or plain
            fractionary reader indices of shape (M,)
        y: array_like
            Elements are `float` with projected y-axis coordinate or plain
            fractionary reader indices of shape (M,)
        z: array_like
            Elements are `float` with depth below the reference level, positive
            up and shape (M,)
        testing: bool
            ????

        Returns
        -------
        variables: dict
            ????
        """
        print('###################################')
        print('#########REQUESTED#################')
        print('requested vars', requested_variables)
        print('time', time)
        print('x', x)
        print('y', y)
        print('z', z)
        # Empty unrotated velocity cache
        for key in list(self._uv_cache.keys()):
            self._uv_cache.pop(key)
        if z is not None:
            # Convert to numpy array
            z =  np.atleast_1d(z)
        variables = {}
        start_time = datetime.now()
        nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = \
            self.nearest_time(time)
        requested_variables, time, x, y, z, outside = self.check_arguments(
            requested_variables, time, x, y, z)
        variables['time'] = nearestTime
        # Find nearest x, y
        variables['x'], variables['y'] = self._get_xy(x, y)
        xs, ys = np.meshgrid(
            variables['x'].flatten(),
            variables['y'].flatten())
        base_slice = (
            slice(ys.min(), ys.max() + 1),
            slice(xs.min(), xs.max() + 1),
        )
        cube_slices = {
            3: (slice(indxTime, indxTime + 1), *base_slice),
            4: (slice(indxTime, indxTime + 1), slice(None), *base_slice),
            5: (
                slice(indxTime, indxTime + 1),
                slice(None),
                slice(None),
                *base_slice,
            ),
        }
        variables['z'], zs_at_sigma = self._get_depth_coords(
            cube_slices[3],
            z,
        )
        for variable in requested_variables:
            invarname = self.standard_variable_mapping[variable]
            try:
                ndim = self.Dataset[invarname].data.ndim
            except KeyError:
                # Handle variables grouped in R1
                ndim = self.Dataset['R1'].data.ndim - 1
            try:
                cube_slice = cube_slices[ndim]
            except KeyError:
                cube_slice = base_slice
            cube, mask = self._get_cube(variable, cube_slice, testing=testing)
            cube = np.squeeze(cube)
            cube = np.atleast_2d(cube.reshape(cube.shape[0], -1))
            if ndim > 3:
                # Send to profile interpolation and reshape it back into an
                # (zs, ys, xs) cube
                variables[variable] = self._interpolate_profile(
                    cube,
                    zs_at_sigma,
                    variables['z'],
                ).reshape(-1, *mask.shape[-2:]).squeeze()
                
            else:
                variables[variable] = cube
        return  variables
