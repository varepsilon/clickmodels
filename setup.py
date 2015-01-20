from distutils.core import setup
import glob

from setuptools import setup

def read_md(file_name):
    try:
        from pypandoc import convert
        return convert(file_name, 'rest')
    except:
        return ''

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
