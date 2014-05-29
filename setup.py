#!/usr/bin/env python
"""NumPy based Continuous Time Markov Chain TREE algorithms

"""

DOCLINES = __doc__.split('\n')

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.
#
# More stuff added for Cython extensions.

from distutils.core import setup

from distutils.extension import Extension

from Cython.Distutils import build_ext

setup(
        name='npctmctree',
        version='0.1',
        description=DOCLINES[0],
        author='alex',
        url='https://github.com/argriffing/npctmctree/',
        download_url='https://github.com/argriffing/npctmctree/',
        packages=['npctmctree'],
        test_suite='nose.collector',
        package_data={'npctmctree' : ['tests/test_*.py']},
        cmdclass={'build_ext' : build_ext},
        ext_modules=[Extension('npctmctree.cyem', ['npctmctree/cyem.pyx'])],
        )

