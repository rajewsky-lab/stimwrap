from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

try:
    from stimwrap import __author__, __email__, __version__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = __version__ = ''

with open('requirements.txt', 'r') as f:
    required_packages = f.read().splitlines()

setup(
    name="stimwrap",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="seerpy: the ultimate success prediction tool.",
    long_description=long_description,
    url="https://github.com/nukappa/stimwrap",
    license='MIT',
    install_requires=required_packages,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X"
    ]
)