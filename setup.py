#!/usr/bin/env python

from setuptools import setup

setup(name='qopt',
      version='0.2',
      description='Quantum Optics tools',
      author='Jonas S. Neergaard-Nielsen',
      author_email='j@neer.dk',
      packages=['qopt'],
      install_requires=[
            'numpy',
            'scipy',
            'matplotlib'
      ]
     )
