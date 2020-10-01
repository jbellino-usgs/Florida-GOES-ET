import sys
from distutils.core import setup


long_description = \
      """
      GOESet is a cobbled-together set of modules to help me get stuff done.
      """

repo_url = 'https://code.usgs.gov/jbellino/goeset'

setup(name="goeset",
      description=long_description,
      long_description=long_description,
      author="Jason Bellino",
      author_email='jbellino@usgs.gov',
      url=repo_url,
      download_url=repo_url + '/-/archive/master/goeset-master.tar.gz',
      license='New BSD',
      platforms='Windows',
      packages=["goeset"],
      version="0.1")
