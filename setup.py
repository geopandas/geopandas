from distutils.core import setup

setup(name='geopandas',
      version='0.1dev',
      description='Geographic pandas extensions',
      license='BSD',
      author='Kelsey Jordahl',
      author_email='kjordahl@enthought.com',
      url='',
      packages=['geopandas'],
      install_requires=['pandas', 'shapely', 'fiona', 'descartes'],
)
