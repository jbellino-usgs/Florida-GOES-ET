import os
import pyproj
import pandas as pd
import numpy as np

ancpth = os.path.join(os.path.dirname(__file__), 'ancillary')
shppth = os.path.join(os.path.dirname(__file__), 'shp')

lcc_wkt = \
    """PROJCS["North_America_Lambert_Conformal_Conic",
    GEOGCS["GCS_North_American_1983",
        DATUM["North_American_Datum_1983",
            SPHEROID["GRS_1980",6378137,298.257222101]],
        PRIMEM["Greenwich",0],
        UNIT["Degree",0.017453292519943295]],
    PROJECTION["Lambert_Conformal_Conic_2SP"],
    PARAMETER["False_Easting",5632642.22547],
    PARAMETER["False_Northing",4612545.65137],
    PARAMETER["Central_Meridian",-107],
    PARAMETER["Standard_Parallel_1",50],
    PARAMETER["Standard_Parallel_2",50],
    PARAMETER["Latitude_Of_Origin",50],
    UNIT["Meter",1],
    AUTHORITY["EPSG","102009"]]"""

nad83_wkt = \
    """GEOGCS["NAD83",
    DATUM["North_American_Datum_1983",
    SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],
    PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]]"""


def fill(data, invalid=None):
    """
    Taken from https://stackoverflow.com/a/9262129/698809

    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    from scipy import ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]


class Pixels(object):
    """

    :param object:
    :return:
    """
    def __init__(self):
        self.nrow, self.ncol = 407, 474
        self._pixel_list = self.load_pixels()

    def load_pixels(self):
        fname = os.path.join(ancpth, 'pixels.txt')
        pix_df = pd.read_csv(fname, index_col=['pixel'], usecols=['pixel', 'latitude', 'longitude'])
        pix_df = pix_df.loc[self.nrpix_sequence_number]

        nad83 = pyproj.Proj(nad83_wkt)
        lcc = pyproj.Proj(lcc_wkt)
        transformer = pyproj.Transformer.from_proj(nad83, lcc)
        _x, _y = transformer.transform(pix_df.longitude.values,
                                       pix_df.latitude.values)

        pix_df.loc[:, 'x'] = _x
        pix_df.loc[:, 'y'] = _y

        i = np.zeros((self.nrow, self.ncol), dtype=int)
        for ii in range(self.nrow):
            i[ii, :] += ii
        j = np.zeros((self.nrow, self.ncol), dtype=int)
        for jj in range(self.ncol):
            j[:, jj] += jj

        pix_df.loc[:, 'i'] = i.ravel()
        pix_df.loc[:, 'j'] = j.ravel()

        pix_df.loc[:, 'fortran_sequence_number'] = self.fortran_sequence_number

        return pix_df

    @property
    def nrpix_sequence_number(self):
        """
        NEXRAD pixel numbering starts at lower-left and increases column-wise.

        """
        seq = np.array(range(1, (self.nrow * self.ncol) + 1))
        seq = seq.reshape(self.nrow, self.ncol)
        return seq[::-1, :].ravel()

    @property
    def fortran_sequence_number(self):
        """
        Fortran sequencing starts at lower-right and increases with column-major order.

        """
        seq = np.array(range(1, (self.nrow * self.ncol) + 1))[::-1]
        seq = seq.reshape(self.nrow, self.ncol, order='f')
        return seq.ravel()

    @property
    def data(self):
        return self._pixel_list

    @property
    def pixel_ids(self):
        return self._pixel_list.index.values

    @property
    def latitude(self):
        return self._pixel_list.latitude.values

    @property
    def longitude(self):
        return self._pixel_list.longitude.values

    @property
    def x(self):
        return self._pixel_list.x.values

    @property
    def y(self):
        return self._pixel_list.y.values


class GoesAsciiFile(object):
    """
    A thin wrapper around the Pandas DataFrame class to help read
    and manipulate legacy ASCII files.

    Attributes
    ----------

    Methods
    -------

    Examples
    --------
    >>> import goeset
    >>> etfile = goeset.GoesAsciiFile('Florida_2017.txt')
    >>> eto = etfile.get_array('RET')

    >>> import goeset
    >>> etfile  = goeset.GoesAsciiFile('Florida_2017.txt')
    >>> df = etfile.get_dataframe(nrows=500)

    """
    def __init__(self, fpth):

        s = f'Could not locate input file {fpth}.'
        assert os.path.isfile(fpth), s
        self.fpth = fpth

        self.nodata_value = -9999.9
        # self.aea_proj4 = '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-84 +x_' \
        #     '0=0 +y_0=0 +datum=NAD83 +units=ft +no_defs '

        self.pixels = Pixels()
        self.reference_table = self.get_pixel_reference_table()
        self.nrow, self.ncol = 407, 474
        self._df = None
        self._dates = None
        # self._x, self._y = self._get_xy_arrays()
        # self._latitude, self._longitude = self._get_latlon_arrays()
        self._oldfmt, self._header = self.get_header()

    @property
    def df(self):
        if self._df is None:
            self._df = self.get_dataframe()
        return self._df

    @property
    def dates(self):
        if self._df is None:
            self._df = self.get_dataframe()
        return self._dates

    @property
    def header(self):
        return self._header

    @property
    def latitude(self):
        return self.pixels.data.latitude.values.reshape(self.nrow, self.ncol)

    @property
    def longitude(self):
        return self.pixels.data.longitude.values.reshape(self.nrow, self.ncol)

    @staticmethod
    def get_pixel_reference_table():
        """
        Load the list of pixels.
        :return:
        """
        dtype = {'NRpix': int, 'fips_county': str}
        fname = os.path.join(ancpth, 'pixel_reference.csv')
        tbl = pd.read_csv(fname, index_col=['NRpix'], dtype=dtype)
        tbl = tbl.sort_index()
        return tbl

    # @staticmethod
    # def get_pixels():
    #     """
    #     Load the list of pixels.
    #     :return:
    #     """
    #     fname = os.path.join(ancpth, 'pixels.txt')
    #     return pd.read_csv(fname, index_col=['pixel'])

    def get_header(self):
        with open(self.fpth, 'r') as f:
            header = f.readline().strip().split()
        if 'YYYYMMDD' in header:
            header = ['YYYYMMDD', 'Lat', 'Lon', 'NRpix', 'PET', 'ETo',
                      'Solar', 'Albedo', 'RHmax', 'RHmin', 'Tmax', 'Tmin',
                      'ws2m']
            oldfmt = False
        else:
            oldfmt = True
            header = ['YYYYMMDD', 'Lat', 'Lon', 'NRpix', 'PET', 'ETo',
                      'Solar', 'RHmax', 'RHmin', 'Tmax', 'Tmin', 'ws2m']
        return oldfmt, header

    def get_dataframe(self, flush=True, **kwargs):
        """

        Parameters
        ----------
        flush : bool
            If true, reload fresh copy of entire pandas.DataFrame

        """
        if self._df is None or flush:
            self._df = self._read_file(**kwargs)
            self._dates = [pd.Timestamp(t) for t in pd.to_datetime(self._df['YYYYMMDD']).unique()]
        return self._df

    def _read_file(self, **kwargs):
        """
        Get data stored in the ASCII file as a pandas.DataFrame.

        Parameters
        ----------
        **kwargs : dictionary
            Keyword arguments passed to pandas.read_csv

        Returns
        -------
        data : pandas.DataFrame

        """

        # Check specified parameter name against available columns
        if 'usecols' in kwargs.keys():
            cols = kwargs['usecols']
            if isinstance(cols, str):
                cols = [cols]
            check = all(item in cols for item in self._header)
            assert check, 'One or more parameters pass in the usecols argument are invalid.'
            usecols = ['YYYYMMDD', 'Lat', 'Lon', 'NRpix']
            for c in cols:
                if c not in usecols:
                    usecols.append(c)

        else:
            usecols = self._header

        if self._oldfmt:
            header = None
            skiprows = 0

        else:
            header = 'infer'
            skiprows = 1

        # Read the file
        dtype = {'YYYYMMDD': str, 'Lat': float, 'Lon': float,
                 'NRpix': int}
        data = pd.read_csv(self.fpth, delim_whitespace=True, dtype=dtype,
                           usecols=usecols, names=usecols, header=header,
                           low_memory=True, skiprows=skiprows, **kwargs)

        # Replace nodata values
        data = data.replace(self.nodata_value, np.nan)

        # Add datetime dates
        data['date'] = pd.to_datetime(data['YYYYMMDD'])

        # Set index
        data = data.set_index(['date', 'NRpix'], drop=False).sort_index()

        return data

    def get_array(self, param, flush=False):
        """
        Get data stored in the ASCII file as a numpy.maskedarray.

        Parameters
        ----------
        param : str
            Name of the parameter for which data will be returned
        flush : bool
            If true, reload fresh copy of data file as a pandas.DataFrame

        Returns
        -------
        array : numpy.maskedarray

        """
        # Check specified parameter name against available columns
        s = f'Specified parameter "{param}" not recognized. Available parameters:\n  ' + '\n  '.join(self._header)
        assert param.lower() in [p.lower() for p in self._header], s
        param = self._header[[p.lower() for p in self._header].index(param.lower())]
        if flush:
            self.get_dataframe(flush=flush)
        if self._df is None:
            self.get_dataframe()

        # Get a dataframe with dates, pixel ID, and data for the specified parameter
        df = self._df[['YYYYMMDD', 'NRpix', param]].copy()

        # Get list of unique dates in this file
        try:
            dates = df.date.unique()
        except AttributeError:
            # Not datetime objects
            dates = df.YYYYMMDD.unique()

        # Set index
        df = df.set_index('NRpix')

        a = list()
        for i, dtstr in enumerate(dates):
            dfi = df[df.YYYYMMDD == dtstr]
            z = np.ones((self.nrow * self.ncol)) * self.nodata_value
            idx = self.pixels.data.index.intersection(dfi.index)
            z[self.pixels.data.loc[idx, 'fortran_sequence_number'] - 1] = dfi.loc[idx, param]
            a.append(z)

        array = np.stack(a)
        array = np.ma.masked_equal(array, self.nodata_value)
        array = array.reshape((len(dates), self.nrow, self.ncol), order='f')
        return array[:, ::-1, ::-1]


# class GoesNetcdfFile(object):
#     import netCDF4
#     """
#         A thin wrapper around the netCDF4.Dataset class to help read
#         and manipulate netcdf files.
#
#         Attributes
#         ----------
#
#         Methods
#         -------
#
#         Examples
#         --------
#         >>> import goeset
#         >>> ncfile = goeset.GoesNetcdfFile('fl.et.2019.v.1.0.nc')
#         >>> eto = ncfile.get_array('ETo')
#
#         """
#     def __init__(self, fpth):
#
#         s = f'Could not locate input file {fpth}.'
#         assert os.path.isfile(fpth), s
#         self.fpth = fpth
#         with netCDF4.Dataset(fpth, 'r') as ncfile:
#             if 'x' in ncfile.variables.keys():
#                 self._x = ncfile.variables['x'][:]
#             else:
#                 self._x = None
#             if 'y' in ncfile.variables.keys():
#                 self._y = ncfile.variables['y'][:]
#             else:
#                 self._y = None
#             self._latitude = ncfile.variables['lat'][:]
#             self._longitude = ncfile.variables['lon'][:]
#             self._strtimes = ncfile.variables['time'][:]
#             self._times = pd.date_range(self._strtimes[0],
#                                         self._strtimes[-1],
#                                         freq='D')
#             self.nrow, self.ncol = len(self._latitude), len(self._longitude)
#             self.nday = self.times.shape
#             self.fill_value = ncfile.variables['Tmin']._FillValue
#             self._names = list(ncfile.variables.keys())
#
#     @property
#     def names(self):
#         return self._names
#
#     @property
#     def times(self):
#         return self._times
#
#     @property
#     def x(self):
#         return self._x
#
#     @property
#     def y(self):
#         return self._y
#
#     @property
#     def latitude(self):
#         return self._latitude
#
#     @property
#     def longitude(self):
#         return self._longitude
#
#     @staticmethod
#     def load_pts():
#         fname = os.path.join(shppth, 'goes_pts.shp')
#         return gpd.read_file(fname)
#
#     def get_array(self, param):
#         s = f'Parameter not found: {param}'
#         with netCDF4.Dataset(self.fpth, 'r') as ncfile:
#             assert param in self.names, s
#             return ncfile.variables[param][:]
#
#     def tabularize(self, params=None):
#         """
#         Tabularize numpy ndarray to legacy ASCII file format.
#
#         Parameters
#         ----------
#         params : str or list of str
#             List of the parameter names to tabularize.
#
#         Returns
#         -------
#         data : pandas.DataFrame
#
#         Examples
#         --------
#         >>> import goeset
#         >>> ncfile = goeset.GoesNetcdfFile('fl.et.2019.v.1.0.nc')
#         >>> tabular_data = ncfile.tabularize()
#         """
#         if params is None:
#             params = ['PET', 'ETo', 'Solar', 'Albedo',
#                       'Tmin', 'Tmax', 'RHmin', 'RHmax', 'ws2m']
#         else:
#             if isinstance(params, str):
#                 params = [params]
#
#         # Copy array data into memory
#         vardata = {}
#         for v in params:
#             vardata[v] = self.get_array(v)
#
#         # Load the NEXRAD pixel point geometry
#         goes_pts = self.load_pts()
#
#         # Begin polling the data
#         data = []
#         i, j = goes_pts.i.values, goes_pts.j.values
#         for k, dt in enumerate(self.times):
#             df = pd.DataFrame(data={'YYYYMMDD': [f'{dt.year}{dt.month:0>2d}{dt.day:0>2d}'] * len(goes_pts),
#                                     'Lat': goes_pts.latitude.values,
#                                     'Lon': goes_pts.longitude.values,
#                                     'NRpix': goes_pts.NRpix.values})
#             for v in params:
#                 df[v] = vardata[v][k, i, j]
#
#             # Copy information we can use to subset the dataframe
#             # later on
#             subset = ['wmd', 'county_cd']
#             for field in subset:
#                 df[field] = goes_pts[field]
#
#             data.append(df)
#         data = pd.concat(data)
#
#         # Fill NaN values with nodata values
#         for p in params:
#             if p in data.columns:
#                 data[p] = data[p].fillna(self.fill_value)
#
#         # Drop rows with no NEXRAD pixel ID and reset column dtype
#         # to integer
#         col = 'NRpix'
#         data = data.dropna(subset=[col])
#         data.loc[:, col] = data.loc[:, col].astype(int)
#
#         # Sort by date and NEXRAD pixel ID
#         data = data.sort_values(['YYYYMMDD', 'NRpix'])
#
#         return data

