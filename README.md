Introduction
-----------------------------------------------

This code repository was developed to provide a programmatic interface with legacy files from the [Florida GOES reference and potential evapotranspiration][1] project which are stored in a tabular ASCII format.

Utilities in this repository allow a user to read the tabular data into a Pandas.DataFrame object or, alternatively, as numpy.ndarray objects. See the example jupyter notebooks in the `goeset/examples/` directory.

Penman-Monteith calculated reference evapotranspiration:

![Penman-Monteith](goeset/img/penman-monteith_ETo.png?raw=true)

Priestley-Taylor calculated potential evapotranspiration:

![Priestley-Taylor](goeset/img/priestley-taylor_PET.png?raw=true)


Installation
-----------------------------------------------

**Python versions:**

GoesET requires **Python** 3.7 (or higher)


**Dependencies:**

GoesET requires **NumPy** 1.18 (or higher), **Pandas** 1.0.5 (or higher), and **netCDF4** 1.4.2 (or higher). GoesET may work with different versions of these packages, however they have not been tested.


**Cloning the repository:**

To clone the repository via SSH to your local machine type:
    
    git clone git@code.usgs.gov:jbellino/florida-goes-et.git
    
To clone the repository via HTTPS to your local machine type:

    git clone https://code.usgs.gov/jbellino/florida-goes-et.git


**Installing from the git repository:**

To install GoesET from the local git repository created in the previous step type:

    pip install -e <path/to/local/repository/.>

**Convert ASCII File to NetCDF**

From the command line, convert a legacy ascii text file with tabularized data into a georeferenced, grid-based NetCDF file:

    python ascii2netcdf.py <input ascii file path--required> <output netcdf file path--optional>

Disclaimer
----------

This software has been approved for release by the U.S. Geological Survey
(USGS). Although the software has been subjected to rigorous review, the USGS
reserves the right to update the software as needed pursuant to further analysis
and review. No warranty, expressed or implied, is made by the USGS or the U.S.
Government as to the functionality of the software and related material nor
shall the fact of release constitute any such warranty. Furthermore, the
software is released on condition that neither the USGS nor the U.S. Government
shall be held liable for any damages resulting from its authorized or
unauthorized use.

[1]: https://www.usgs.gov/centers/car-fl-water/science/reference-and-potential-evapotranspiration
