"""Setup of gan-face-editing."""

from setuptools import find_packages, setup

# Installation
config = {
    'name': 'sagan-facies-modeling',
    'version': '0.2.2',
    'description': 'Facies modeling with SAGAN.',
    'author': 'Valentin Goldite',
    'author_email': 'valentin.goldite@gmail.com',
    'packages': find_packages(),
    'zip_safe': True
}

setup(**config)
