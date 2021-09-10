import os
from goeset import GoesAsciiFile


# Test GoesAsciiFile
loadpth = os.path.join(os.path.dirname(__file__), '..', 'examples', 'data')
fname = os.path.join(loadpth, 'sample_data.txt')
asciifile = GoesAsciiFile(fname)

assert len(asciifile.dates) == 10, 'Length of dates in GoesAsciiFile does not match expected shape.'

shape = (10, 407, 474)
a = asciifile.get_array('ETo')
assert a.shape == shape, 'GoesAsciiFile array does not match expected shape.'

a = asciifile.get_array('PET')
assert a.shape == shape, 'GoesAsciiFile array does not match expected shape.'

year = 2017
fname = os.path.join(loadpth, f'Florida_{year}.txt')
asciifile = GoesAsciiFile(fname)
n = 20
assert len(asciifile.get_dataframe(nrows=n)) == n, 'Length of DataFrame does not match expected.'
