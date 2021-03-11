import os
import sys
import netCDF4
import pandas as pd
import geopandas as gpd
from datetime import datetime


def main():
    """
    This utility converts the input GOES ET NetCDF file into the
    legacy ASCII file format.

    Arguments
    ---------
    input file : str
        Name of the input NetCDF file to process.
    output file: str (optional)
        Name of the output ASCII file. If not specified, the output
        file name will be generated from the input file name.

    Examples
    --------
    >>> python netcdf2ascii.py fl.et.2019.v.0.1.nc

    """

    # Parse arguments
    narg = len(sys.argv)
    assert 1 < narg <= 3, 'Arguments not recognized. Specify <input file> [output file].'

    if narg == 2:
        input_f = sys.argv[1]
        pre, ext = os.path.splitext(input_f)
        output_f = pre + '.txt'
    elif narg == 3:
        input_f, output_f = sys.argv[1:]

    assert os.path.isfile(input_f), f'File not found {os.path.abspath(input_f)}.'
    print(f'{input_f} -> {output_f}')

    if os.path.isfile(output_f):
        print(f'Removing existing file {output_f}')
        os.remove(output_f)

    # Load shapefile containing all GOES points
    # for this dataset
    print('Reading spatial information')
    shppth = os.path.join(os.path.dirname(__file__), 'shp')
    fname = os.path.join(shppth, 'goes_pts.shp')
    goes_pts = gpd.read_file(fname)
    lat = goes_pts.latitude.values
    lon = goes_pts.longitude.values
    nrpix = goes_pts.NRpix.values
    i = goes_pts.i.values
    j = goes_pts.j.values

    data = list()

    # Read the NetCDF file
    ignore = ['lat', 'lon', 'time', 'x', 'y', 'Lambert_Conformal', 'SolarCode']
    with netCDF4.Dataset(input_f, 'r') as src:

        # Read dates
        dates = src.variables['time'][:]
        dates = [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                 for dt in dates]

        # Read all data arrays
        vardata = {}
        for name, var in src.variables.items():
            if name in ignore:
                continue
            vardata[name] = var[:].copy()

    # Dictionary to convert new variable names to legacy names
    name_dict = {'ETo': 'RET', 'PET': 'PET', 'Solar': 'RS',
                 'Albedo': 'Albedo', 'RHmax': 'Rhmax',
                 'RHmin': 'Rhmin', 'Tmax': 'Tmax', 'Tmin': 'Tmin',
                 'ws2m': 'Ws', 'WMD': 'WMD', 'FIPS': 'FIPS'}
    for k, dt in enumerate(dates):
        # print('Processing', f'{dt.year}-{dt.month:0>2d}-{dt.day:0>2d}')
        d = {'YYYYMMDD': [f'{dt.year}{dt.month:0>2d}{dt.day:0>2d}'] * len(goes_pts),
             'Lat': lat,
             'Lon': lon,
             'NRpix': nrpix}
        if 'WMD' in vardata.keys():
            d['WMD'] = vardata['WMD']
        if 'FIPS' in vardata.keys():
            d['FIPS'] = vardata['FIPS']
        for name, var in src.variables.items():
            if name in ignore:
                continue
            if len(vardata[name].shape) == 3:
                d[name_dict[name]] = vardata[name][k, i, j]
            elif len(vardata[name].shape) == 2:
                d[name_dict[name]] = vardata[name][i, j]
        data.append(pd.DataFrame(d))

    data = pd.concat(data)

    # Set fill null values
    ignore = ['YYYYMMDD', 'Lat', 'Lon', 'NRpix']

    for column in data.columns:
        if column in ignore:
            continue
        data[column] = data[column].fillna(-9999)

    # Drop points that are not defined in the dataset, i.e. over water, etc.
    data = data.dropna(subset=['NRpix'])

    # Ensure pixel numbers are stored as integers
    intcols = ['NRpix', 'WMD', 'FIPS']
    for c in intcols:
        data[c] = data[c].astype(int)

    # Sort the data by date and pixel ID
    data = data.sort_values(['YYYYMMDD', 'NRpix'])

    # Save the output
    print(f'Writing output to {output_f}')
    data.to_csv(output_f, index=False, sep='\t', float_format='%.3f')

    return


if __name__ == '__main__':
    main()
