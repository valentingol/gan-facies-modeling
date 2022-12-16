"""Setup of gan-face-editing."""

import os
import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent

# The text of the README file
README = os.path.join(HERE, "README.md")

# Installation
config = {
    'name': 'gan-facies',
    'version': '2.5.1',
    'description': 'Facies modeling with GAN.',
    'long_description': README,
    'long_description_content_type': 'text/markdown',
    'url': "https://github.com/valentingol/gan-facies-modeling",
    'author': 'Valentin Goldite',
    'author_email': 'valentin.goldite@gmail.com',
    'packages': ['gan_facies'],
    'package_dir': {'gan_facies': 'gan_facies'},
}

setup(**config)
