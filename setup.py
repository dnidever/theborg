#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='theborg',
      version='1.0.2',
      description='Artificial Neural Network routines',
      author='David Nidever, Yuan-Sen Ting',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/theborg',
      packages=find_packages(exclude=["tests"]),
      #scripts=['bin/doppler'],
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)']
      #include_package_data=True,
)
