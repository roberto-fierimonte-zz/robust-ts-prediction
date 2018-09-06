from setuptools import find_packages, setup


with open('requirements.txt') as f:
    requirements = [r.strip('\n') for r in f.readlines()]

setup(
    name='robust-ts-prediction',
    version='0.1.0',
    description='The TS modelling code base',
    url='https://github.com/roberto-fierimonte/robust-ts-prediction',
    author='Roberto Fierimonte',
    packages=find_packages(),
    zip_safe=False,
    install_requires=requirements
)