# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()


setup(
    name='3Dgan',
    description='3D convolutiona GAN for fast calorimeter simulation',
    long_description=readme,
    author='Sofia Vallecorsa',
    author_email='Sofia.Vallecorsa@cern.ch',
    url='https://github.com/svalleco/3Dgan',
    #license=license,
    packages=find_packages(exclude=('docs'))
)
