"""
setup.py 

Setup package for completeness of the model.

Star Tracker Measurement Model
"""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    readme_file = f.read()

with open("LICENSE", encoding="utf-8") as f:
    license_file = f.read()

setup(
    name="Star Tracker Measurement Model",
    version="0.1.0",
    description="Measurement process model for understanding how errors"
    "propagate through the star tracker and its effects on accuracy and precision.",
    long_description=readme_file,
    author="Gagandeep Thapar",
    url="https://github.com/gagandeepthapar/StarTrackerModel",
    license=license_file,
    packages=find_packages(exclude=()),
)
