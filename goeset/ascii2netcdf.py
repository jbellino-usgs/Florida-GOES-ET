import os
import sys
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from glob import glob
from collections import OrderedDict
from datetime import datetime
from goeset import GoesAsciiFile


def main():
    """
    This utility converts the input GOES ET ASCII file into NetCDF
    file format.

    Arguments
    ---------
    input file : str
        Name of the input ASCII file to process.
    output file: str (optional)
        Name of the output NetCDF file. If not specified, the output
        file name will be generated from the input file name.

    Examples
    --------
    >>> python netcdf2ascii.py Florida_2018.txt

    """
    # Parse arguments
    narg = len(sys.argv)
    assert 1 < narg <= 3, 'Arguments not recognized. Specify <input file> [output file].'

    if narg == 2:
        input_f = sys.argv[1]
        pre, ext = os.path.splitext(input_f)
        output_f = pre + '.nc'
    elif narg == 3:
        input_f, output_f = sys.argv[1:]

    assert os.path.isfile(input_f), f'File not found {os.path.abspath(input_f)}.'
    print(f'{input_f} -> {output_f}')

    if os.path.isfile(output_f):
        print(f'Removing existing file {output_f}')
        os.remove(output_f)

    # Load shapefile containing all GOES points
    # for this dataset
    shppth = os.path.join(os.path.dirname(__file__), 'shp')
    fname = os.path.join(shppth, 'goes_pts.shp')
    goes_pts = gpd.read_file(fname)
    latarr = goes_pts.latitude.values
    lonarr = goes_pts.longitude.values

    nrow = len(np.unique(latarr))
    ncol = len(np.unique(lonarr))

    asciifile = GoesAsciiFile(input_f)

    # Get tabular data
    df = asciifile.get_dataframe()

    dates = np.array([datetime.strptime(dstr, '%Y%m%d') for dstr in df.YYYYMMDD.unique()])

    with nc.Dataset(output_f, mode='w') as ncfile:
        title = f'Automatically generated from input file {os.path.abspath(input_f)}'
        ncfile.title = title
        ncfile.createDimension('lat', nrow)  # latitude axis
        ncfile.createDimension('lon', ncol)  # longitude axis
        ncfile.createDimension('time', len(dates))  # unlimited axis (can be appended to).

        # Define two variables with the same names as dimensions,
        # a conventional way to define "coordinate variables".
        lat = ncfile.createVariable('lat',
                                    np.float32,
                                    ('lat',),
                                    zlib=True,
                                    complevel=4,
                                    least_significant_digit=None)
        lat[:] = np.unique(latarr)

        lon = ncfile.createVariable('lon',
                                    np.float32,
                                    ('lon',),
                                    zlib=True,
                                    complevel=4,
                                    least_significant_digit=None)
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        lon[:] = np.unique(lonarr)

        # Store times
        time = ncfile.createVariable('time',
                                     np.int32,
                                     ('time',),
                                     zlib=True,
                                     complevel=4,
                                     least_significant_digit=None)
        origin_str = '1985-1-1 00:00:00'
        origin_dt = datetime.strptime(origin_str, '%Y-%m-%d %H:%M:%S')
        times = np.array([tdelta.days for tdelta in dates - origin_dt])
        time.units = f'days since {origin_str}'
        time.long_name = f'Time, in days since {origin_str}'
        time.var_desc = f'Time, in days since {origin_str}'
        time[:] = times

        # Store dates
        date = ncfile.createVariable('date',
                                     str,
                                     ('time',),
                                     zlib=True,
                                     complevel=4,
                                     least_significant_digit=None)
        date.units = 'YYYYMMDD'
        date.long_name = 'ASCII date string: 4-digit year, 2-digit month, 2-digit day'
        for idx, dt in enumerate(dates):
            date[idx] = str(dt.date())

        ignore = ['YYYYMMDD', 'Lat', 'Lon', 'NRpix']
        for c in asciifile.header:
            if c in ignore:
                continue
            p = ncfile.createVariable(c,
                                      np.float32,
                                      ('time', 'lat', 'lon'),
                                      zlib=True,
                                      complevel=4,
                                      least_significant_digit=None,
                                      fill_value=asciifile.nodata_value)
            p[:, :, :] = asciifile.get_array(c)


if __name__ == '__main__':
    main()
