# -*- coding: utf-8 -*-

"""
@author: Tyler Blume
"""


# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TimeMurmur",
    version="0.1.1",
    author="Tyler Blume",
    # url="https://github.com/tblume1992/ThymeBoost",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description = "Time series forecasting at scale with LightGBM",
    author_email = 'tblume@mail.USF.edu', 
    keywords = ['forecasting', 'time series', 'lightgbm'],
      install_requires=[           
                        'numpy',
                        'pandas',
                        'statsmodels',
                        'scikit-learn',
                        'optuna',
                        'scipy',
                        'matplotlib',
                        'lightgbm',
                        'catboost',
                        'thymeboost',
                        'shap'
                        ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)