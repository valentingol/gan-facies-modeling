"""Setup of gan-face-editing."""

import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Installation
config = {
    'name': 'gan-facies',
    'version': '2.3.0',
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
