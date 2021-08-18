import os
import netCDF4
import pandas as pd
import geopandas as gpd
import numpy as np

ancpth = os.path.join(os.path.dirname(__file__), 'ancillary')
shppth = os.path.join(os.path.dirname(__file__), 'shp')


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
        self.aea_proj4 = '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-84 +x_' \
            '0=0 +y_0=0 +datum=NAD83 +units=ft +no_defs '

        self.pixels = self.get_pixels()
        self.nrow, self.ncol = 407, 474
        self._df = None
        self._dates = None
        # self._x, self._y = self._get_xy_arrays()
        # self._latitude, self._longitude = self._get_latlon_arrays()
        self._oldfmt, self._header = self.get_header()

    @property
    def df(self):
        return self._df

    @property
    def dates(self):
        return self._dates

    @property
    def header(self):
        return self._header

    @staticmethod
    def get_pixels():
        """
        Load the list of pixels.
        :return:
        """
        dtype = {'NRpix': int, 'fips_county': str}
        fname = os.path.join(ancpth, 'pixel_reference.csv')
        pixels = pd.read_csv(fname, index_col=['NRpix'], dtype=dtype)
        pixels = pixels.sort_index()
        return pixels

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
            self._dates = pd.to_datetime(self._df['YYYYMMDD'].unique())
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
        s = f'Specified parameter "{param}" not recognized.'
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
            dates = df.YYYYMMDD.dt.strftime('%Y-%m-%d').unique()
        except AttributeError:
            # Not datetime objects
            dates = df.YYYYMMDD.unique()

        # Set date index
        df = df.set_index('NRpix')

        a = list()
        for i, dtstr in enumerate(dates):
            dfi = df[df.YYYYMMDD == dtstr]
            z = np.ones((self.nrow * self.ncol)) * self.nodata_value
            idx = self.pixels.index.intersection(dfi.index)
            z[self.pixels.loc[idx, 'sequence_number'] - 1] = dfi.loc[idx, param]
            a.append(z)

        array = np.stack(a)
        array = np.ma.masked_equal(array, self.nodata_value)
        array = array.reshape((len(dates), self.nrow, self.ncol), order='f')
        return array[:, ::-1, ::-1]


class GoesNetcdfFile(object):
    """
        A thin wrapper around the netCDF4.Dataset class to help read
        and manipulate netcdf files.

        Attributes
        ----------

        Methods
        -------

        Examples
        --------
        >>> import goeset
        >>> ncfile = goeset.GoesNetcdfFile('fl.et.2019.v.1.0.nc')
        >>> eto = ncfile.get_array('ETo')

        """
    def __init__(self, fpth):

        s = f'Could not locate input file {fpth}.'
        assert os.path.isfile(fpth), s
        self.fpth = fpth
        with netCDF4.Dataset(fpth, 'r') as ncfile:
            if 'x' in ncfile.variables.keys():
                self._x = ncfile.variables['x'][:]
            else:
                self._x = None
            if 'y' in ncfile.variables.keys():
                self._y = ncfile.variables['y'][:]
            else:
                self._y = None
            self._latitude = ncfile.variables['lat'][:]
            self._longitude = ncfile.variables['lon'][:]
            self._strtimes = ncfile.variables['time'][:]
            self._times = pd.date_range(self._strtimes[0],
                                        self._strtimes[-1],
                                        freq='D')
            self.nrow, self.ncol = len(self._latitude), len(self._longitude)
            self.nday = self.times.shape
            self.fill_value = ncfile.variables['Tmin']._FillValue
            self._names = list(ncfile.variables.keys())

    @property
    def names(self):
        return self._names

    @property
    def times(self):
        return self._times

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    def load_pts(self):
        fname = os.path.join(shppth, 'goes_pts.shp')
        return gpd.read_file(fname)

    def get_array(self, param):
        s = f'Parameter not found: {param}'
        with netCDF4.Dataset(self.fpth, 'r') as ncfile:
            assert param in self.names, s
            return ncfile.variables[param][:]

    def tabularize(self, params=None):
        """
        Tabularize numpy ndarray to legacy ASCII file format.

        Parameters
        ----------
        params : str or list of str
            List of the parameter names to tabularize.

        Returns
        -------
        data : pandas.DataFrame

        Examples
        --------
        >>> import goeset
        >>> ncfile = goeset.GoesNetcdfFile('fl.et.2019.v.1.0.nc')
        >>> tabular_data = ncfile.tabularize()
        """
        if params is None:
            params = ['PET', 'ETo', 'Solar', 'Albedo',
                      'Tmin', 'Tmax', 'RHmin', 'RHmax', 'ws2m']
        else:
            if isinstance(params, str):
                params = [params]

        # Copy array data into memory
        vardata = {}
        for v in params:
            vardata[v] = self.get_array(v)

        # Load the NEXRAD pixel point geometry
        goes_pts = self.load_pts()

        # Begin polling the data
        data = []
        i, j = goes_pts.i.values, goes_pts.j.values
        for k, dt in enumerate(self.times):
            df = pd.DataFrame(data={'YYYYMMDD': [f'{dt.year}{dt.month:0>2d}{dt.day:0>2d}'] * len(goes_pts),
                                    'Lat': goes_pts.latitude.values,
                                    'Lon': goes_pts.longitude.values,
                                    'NRpix': goes_pts.NRpix.values})
            for v in params:
                df[v] = vardata[v][k, i, j]

            # Copy information we can use to subset the dataframe
            # later on
            subset = ['wmd', 'county_cd']
            for field in subset:
                df[field] = goes_pts[field]

            data.append(df)
        data = pd.concat(data)

        # Fill NaN values with nodata values
        for p in params:
            if p in data.columns:
                data[p] = data[p].fillna(self.fill_value)

        # Drop rows with no NEXRAD pixel ID and reset column dtype
        # to integer
        col = 'NRpix'
        data = data.dropna(subset=[col])
        data.loc[:, col] = data.loc[:, col].astype(int)

        # Sort by date and NEXRAD pixel ID
        data = data.sort_values(['YYYYMMDD', 'NRpix'])

        return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ncf = GoesNetcdfFile(r'examples\data\fl.et.2019.v.1.0.nc')
    # ncf.tabularize(fout='test.txt', wmd='swf')

    # asciifile = GoesAsciiFile('test.txt')
    # a = asciifile.get_array('ETo')
    # plt.imshow(a[0])
    # plt.show()

    ncf = GoesNetcdfFile(r'examples\data\fl.et.2019.v.1.0.nc')
    tab = ncf.tabularize()
    print(tab.head())

    # asciifile = GoesAsciiFile('test_wmd.txt')
    # a = asciifile.get_array('ETo')
    # plt.imshow(a[0])
    # plt.show()

    # year = 2018
    # fpth = fr'P:\E2H11_9BD00_StatewideET\Products\GOES_ET\Florida_{year}\Florida_{year}.txt'
    # asciifile = GoesAsciiFile(fpth)
    # print(asciifile.get_dataframe(nrows=5))
    # fpth = 'test.txt'
    # asciifile = GoesAsciiFile(fpth)
    # print(asciifile.get_dataframe(nrows=5))



