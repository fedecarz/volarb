from setuptools import setup, find_packages

setup(
    name="volarb",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "yfinance",
        "matplotlib",
    ],
)