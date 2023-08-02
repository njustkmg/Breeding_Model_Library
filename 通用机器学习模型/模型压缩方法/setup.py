

import os

from setuptools import find_packages, setup

root = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as fp:
    install_requirements = fp.readlines()

# read official README.md
with open('README.md', encoding='utf8') as fp:
    long_desc = fp.read()

version = "0.1.0"

setup(name='model_compression',
      version=version,
      description='Model Compression',
      long_description=long_desc,
      long_description_content_type='text/markdown',
      python_requires='>=3.6',
      packages=find_packages(),
      install_requires=install_requirements,
      include_package_data=True,
      )
