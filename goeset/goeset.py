import os
import pyproj
import pandas as pd
import numpy as np
from datetime import datetime

ancpth = os.path.join(os.path.dirname(__file__), 'ancillary')


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

        self.dtype_dict = {'YYYYMMDD': str, 'Lat': float, 'Lon': float,
                           'NRpix': int, 'PET': float, 'RET': float,
                           'RS': float, 'Albedo': float, 'Rhmax': float,
                           'Rhmin': float, 'Tmax': float, 'Tmin': float,
                           'Ws': float}

        self.pixels = self.get_pixels()
        self.nrow, self.ncol = 407, 474
        self._df = None
        self._dates = None
        self._x, self._y = self._get_xy_arrays()
        self._latitude, self._longitude = self._get_latlon_arrays()

    @property
    def df(self):
        return self._df

    @property
    def dates(self):
        return self._dates

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

        # Determine whether the file has new format or old
        with open(self.fpth, 'r') as f:
            header = f.readline()

        # ignore = ('Lat', 'Lon', 'latitude', 'longitude')
        ignore = []

        newfmt = False
        if 'YYYYMMDD' in header:
            newfmt = True

        # Check specified parameter name against available columns
        if newfmt:
            dtype = self.dtype_dict.copy()
            names = None
            header = 'infer'
            dtype = {k: v for (k, v) in dtype.items() if k not in ignore}
            usecols = [c for c in dtype.keys() if c not in ignore]
        else:
            dtype = self.dtype_dict.copy()
            dtype.pop('Albedo')
            header = None
            names = list(dtype.keys())
            dtype = {k: v for (k, v) in dtype.items() if k not in ignore}
            usecols = [names.index(c) for c in dtype.keys() if c not in ignore]
            names = [n for n in names if n not in ignore]

        # if usecols is not None and 'YYYYMMDD' not in usecols:
        #     usecols = usecols.insert(0, 'YYYYMMDD')

        # Read the file
        date_parser = lambda x: datetime.strptime(x, "%Y%m%d")
        data = pd.read_csv(self.fpth, delim_whitespace=True, dtype=dtype, names=names,
                           usecols=usecols, header=header, low_memory=True,
                           parse_dates=['YYYYMMDD'], date_parser=date_parser, **kwargs)

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
        s = f'Specified parameter {param} not recognized.'
        ignore = ['YYYYYMMDD', 'Lat', 'Lon', 'NRpix']
        pstr = ', '.join([p for p in self.dtype_dict.keys() if p not in ignore])
        s += f' Please choose from the following parameters: {pstr}.'
        assert param.lower() in [d.lower() for d in self.dtype_dict.keys()], s
        if flush:
            self.get_dataframe(flush=flush)
        if self._df is None:
            self.get_dataframe()

        # Get a dataframe with dates, pixel ID, and data for the specified parameter
        df = self._df[['YYYYMMDD', 'NRpix', param]].copy()

        # Get list of unique dates in this file
        dates = df.YYYYMMDD.dt.strftime('%Y-%m-%d').unique()

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

    @staticmethod
    def _get_latlon_arrays():
        """
        Get 2-d arrays of latitude and longitude values.
        :return:
        """
        fpth = os.path.join(ancpth, 'latitude.ref')
        latarr = np.loadtxt(fpth)
        fpth = os.path.join(ancpth, 'longitude.ref')
        lonarr = np.loadtxt(fpth)
        return latarr, lonarr

    def _get_xy_arrays(self):
        latarr, lonarr = self._get_latlon_arrays()
        proj = pyproj.Proj(self.aea_proj4)
        return proj(lonarr, latarr)


# def compute_statistics(fpth, statistic='mean', period='monthly'):
#     """
#     Computes statistics on a data file and returns a pandas.DataFrame
#     indexed on NEXRAD pixel ID with a multiindexed column structure using
#     statistic period (e.g. month, year, etc.) and parameter names.
#     :param fpth:
#     :param statistic:
#     :param period:
#     :return:
#     """
#     valid_stats = ['min', 'mean', 'median', 'max', 'std']
#     s = f'Please specify a valid statistic: {", ".join(valid_stats)}'
#     assert statistic.lower() in valid_stats, s
#
#     valid_periods = ['daily', 'monthly', 'annual']
#     s = f'Please specify a valid period for analysis: ' \
#         f'{", ".join(valid_periods)}'
#     assert period in valid_periods, s
#
#     # Determine whether the file has new format or old
#     with open(fpth, 'r') as f:
#         header = f.readline()
#
#     newfmt = False
#     if 'YYYYMMDD' in header:
#         newfmt = True
#
#     # Check specified parameter name against available columns
#     if newfmt:
#         datecol = 'YYYYMMDD'
#         pixcol = 'NRpix'
#         ignore = ['YYYYMMDD', 'Lat', 'Lon', 'NRpix']
#     else:
#         allcols = dtype_dict2.keys()
#         datecol = 'date'
#         pixcol = 'pixel'
#         ignore = ['date', 'latitude', 'longitude', 'pixel']
#
#     # Read the file
#     data = read_file(fpth, parse_dates=[datecol], nodata=nodata_value)
#
#     df = data.set_index(datecol)
#     if period == 'annual':
#         pers = df.index.year.unique()
#         idxlst = [f'{year}' for year in pers]
#     elif period == 'monthly':
#         # NOTE: Assumes only 1 year per file
#         years = df.index.year.unique()
#         s = 'Error: This utility assumes each data file contains only 1 year' \
#             ' of data.'
#         assert len(years) == 1, s
#         year = years[0]
#         pers = df.index.month.unique()
#         idxlst = [f'{year}-{mo:0>2d}' for mo in pers]
#     elif period == 'daily':
#         idxlst = list(df.index.unique())
#
#     stats = {}
#     statcols = [c for c in df.columns if c not in ignore]
#     for idx in idxlst:
#         perdf = df[idx]
#         gp = perdf.groupby([pixcol])[statcols]
#
#         if statistic == 'min':
#             stats[idx] = gp.min()
#         elif statistic == 'mean':
#             stats[idx] = gp.mean()
#         elif statistic == 'median':
#             stats[idx] = gp.medin()
#         elif statistic == 'max':
#             stats[idx] = gp.max()
#         elif statistic == 'std':
#             stats[idx] = gp.std()
#     stats = pd.concat(stats, axis=1)
#
#     # concatentate column names if tuples from a multiindex column structure
#     newcols = []
#     for c in stats.columns:
#         if len(c) > 1:
#             newc = '_'.join([i for i in c])
#             newcols.append(newc)
#         else:
#             newcols.append(c)
#     stats.columns = newcols
#
#     return stats
