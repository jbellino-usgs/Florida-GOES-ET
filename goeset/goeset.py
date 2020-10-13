import os
import netCDF4
import pyproj
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime

ancpth = os.path.join(os.path.dirname(__file__), 'ancillary')
shppth = os.path.join(os.path.dirname(__file__), 'shp')


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

        dtype = {'YYYYMMDD': str, 'Lat': float, 'Lon': float,
                 'NRpix': int}
        usecols = list(dtype.keys())

        # Determine whether the file has new format or old
        with open(self.fpth, 'r') as f:
            header = f.readline()

        oldfmt = True
        if 'YYYYMMDD' in header:
            oldfmt = False

        # Check specified parameter name against available columns
        if oldfmt:
            params = ['PET', 'RET', 'RS', 'Rhmax', 'Rhmin',
                      'Tmax', 'Tmin', 'Ws']
            if 'usecols' in kwargs.keys():
                usecols += usecols
            # dtype += {'PET': float, 'RET': float,
            #           'RS': float, 'Rhmax': float, 'Rhmin': float,
            #           'Tmax': float, 'Tmin': float, 'Ws': float}
            else:
                usecols += params
            header = None
            skiprows = 0
            names = usecols

        else:
            if 'usecols' in kwargs.keys():
                usecols += usecols
            else:
                usecols += [c for c in header.split() if c not in usecols]
            names = usecols
            header = 'infer'
            skiprows = 1

        # Read the file
        date_parser = lambda x: datetime.strptime(x, "%Y%m%d")
        data = pd.read_csv(self.fpth, delim_whitespace=True, dtype=dtype, names=names,
                           header=header, low_memory=True, skiprows=skiprows,
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

        # Determine whether the file has new format or old
        with open(self.fpth, 'r') as f:
            header = f.readline()

        oldfmt = True
        if 'YYYYMMDD' in header:
            oldfmt = False

        # Check specified parameter name against available columns
        if oldfmt:
            params = ['PET', 'RET', 'RS', 'Rhmax', 'Rhmin',
                     'Tmax', 'Tmin', 'Ws']
        else:
            ignore = ['YYYYMMDD', 'Lat', 'Lon', 'NRPIX']
            params = [p for p in header.split() if p not in ignore]

        s = f'Specified parameter {param} not recognized.'
        pstr = ', '.join(params)
        s += f' Please choose from the following parameters: {pstr}.'
        assert param.lower() in [p.lower() for p in params], s
        param = params[[p.lower() for p in params].index(param.lower())]
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
            self._x = ncfile.variables['x'][:]
            self._y = ncfile.variables['y'][:]
            self._latitude = ncfile.variables['lat'][:]
            self._longitude = ncfile.variables['lon'][:]
            self._strtimes = ncfile.variables['time'][:]
            self._times = pd.date_range(self._strtimes[0],
                                        self._strtimes[-1],
                                        freq='D')
            self.nrow, self.ncol = self.x.shape
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

    def get_variable(self, v):
        with netCDF4.Dataset(self.fpth, 'r') as ncfile:
            return ncfile.variables[v][:]

    def load_pts(self):
        fname = os.path.join(shppth, 'goes_pts.shp')
        return gpd.read_file(fname)

    def get_array(self, param):
        s = f'Parameter not found: {param}'
        with netCDF4.Dataset(self.fpth, 'r') as ncfile:
            assert param in ncfile.variables.keys(), s
            return ncfile.variables[param][:]

    def tabularize(self, fout=None, vars=None, wmd=None, county=None):
        """
        Tabularize numpy ndarray to legacy ASCII file format.

        Parameters
        ----------
        fout : str (optional)
            Name of the desired output file.
        vars : str or list of str
            List of the parameter names to tabularize.
        wmd : str or list of str
            List of water management district abbreviations
            to subset the data by.
        county : str or list of str
            List of county FIPS codes to subset the data by.

        Returns
        -------
        data : pandas.DataFrame

        Examples
        --------
        >>> import goeset
        >>> ncfile = goeset.GoesNetcdfFile('fl.et.2019.v.1.0.nc')
        >>> ncfile.tabularize('subset.txt', wmd=['SWFWMD', 'SRWMD'])
        """
        subset = {}
        if wmd is not None:
            if isinstance(wmd, str):
                wmd = [wmd]
            wmd = [w.upper() for w in wmd]
            subset['wmd'] = wmd
        if county is not None:
            if isinstance(county, str):
                county = [county]
            county = [int(c).zfill(3) for c in county]
            subset['county_cd'] = county

        # Make sure arrays have been loaded
        if vars is None:
            vars = ['PET', 'ETo', 'Solar', 'Albedo',
                    'Tmin', 'Tmax', 'RHmin', 'RHmax', 'ws2m']
        else:
            if isinstance(vars, str):
                vars = [vars]
        vardata = {}
        for v in vars:
            vardata[v] = self.get_variable(v)
        goes_pts = self.load_pts()
        data = []
        i, j = goes_pts.i.values, goes_pts.j.values
        for k, dt in enumerate(self.times):
            df = pd.DataFrame(data={'YYYYMMDD': [f'{dt.year}{dt.month:0>2d}{dt.day:0>2d}'] * len(goes_pts),
                                    'Lat': goes_pts.latitude.values,
                                    'Lon': goes_pts.longitude.values,
                                    'NRpix': goes_pts.NRpix.values})
            for v in vars:
                df[v] = vardata[v][k, i, j]

            if subset:
                for k in subset.keys():
                    if hasattr(goes_pts, k):
                        df[k] = goes_pts[k]

            data.append(df)

        data = pd.concat(data)
        data.ETo = data.ETo.fillna(-9999)
        data.PET = data.PET.fillna(-9999)
        data = data.dropna(subset=['NRpix'])
        data.NRpix = data.NRpix.astype(int)
        data = data.sort_values(['YYYYMMDD', 'NRpix'])

        subset_data = []
        if subset:
            for k, v in subset.items():
                for vi in v:
                    subset_data.append(data[data[k] == vi])
        data = pd.concat(subset_data)

        if fout is not None:
            data.to_csv(fout, index=False, sep='\t', float_format='%9.3f')
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
    ncf.tabularize(fout='test_wmd.txt', wmd=['SRWMD', 'SWFWMD'])

    asciifile = GoesAsciiFile('test_wmd.txt')
    a = asciifile.get_array('ETo')
    plt.imshow(a[0])
    plt.show()

    # year = 2018
    # fpth = fr'P:\E2H11_9BD00_StatewideET\Products\GOES_ET\Florida_{year}\Florida_{year}.txt'
    # asciifile = GoesAsciiFile(fpth)
    # print(asciifile.get_dataframe(nrows=5))
    # fpth = 'test.txt'
    # asciifile = GoesAsciiFile(fpth)
    # print(asciifile.get_dataframe(nrows=5))



