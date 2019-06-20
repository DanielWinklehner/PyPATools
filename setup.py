from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(name='PyPATools',
      version='0.0.1',
      python_requires='>=3',
      description='Several useful classes for simulating particle accelerators',
      url='https://github.com/DanielWinklehner/PyPATools',
      author='Daniel Winklehner',
      author_email='winklehn@mit.edu',
      license='MIT',
      packages=['PyPATools'],
      package_data={'PyPATools': ['Settings.txt', 'Doc/*', 'Examples/*']},
      include_package_data=True,
      ext_modules=cythonize("PyPATools/particles.pyx"),
      include_dirs=[numpy.get_include()],
      zip_safe=False,
      )
