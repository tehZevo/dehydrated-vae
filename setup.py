from setuptools import setup, find_packages

setup(name='dehydrated_vae',
  version='0.0.0',
  install_requires = [
    'keras',
    'numpy',
  ],
  packages=find_packages())
