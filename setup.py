from distutils.core import setup
import glob

from setuptools import setup
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print('Warning: pypandoc module not found, could not convert Markdown to RST')
    read_md = lambda f: open(f, 'r').read()

setup(
    name='clickmodels',
    version='1.0.2',
    author='Aleksandr Chuklin',
    packages=['clickmodels'],
    scripts=glob.glob('bin/*.py'),
    url='https://github.com/varepsilon/clickmodels',
    license='LICENSE',
    description='Probabilistic models of user behavior on a search engine result page',
    long_description=read_md('README.md'),
    install_requires=[],
)
