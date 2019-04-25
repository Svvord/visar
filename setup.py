#!/usr/bin/env python
#-*- coding:utf-8 -*-

with open("README.md", "r") as f:
    long_description = f.read()

from setuptools import setup, find_packages

setup(
    name = "visar",
    version = "0.1.5.2",
    keywords = ("Chemoinformatics", "neural network", "visualized structure-activity relationship", "chemical landscape"),
    description = "This project aims to train neural networks by compound-protein interactions and provides interpretation of the learned model by interactively showing transformed chemical landscape and visualized SAR for chemicals of interest.",
    long_description = long_description,
    long_description_content_type='text/markdown',
    license = "MIT Licence",

    url = "https://github.com/Svvord/visar",
    author = "Qingyang Ding",
    author_email = "dingqy14@mails.tsinghua.edu.cn",
    maintainer = "S. Hou",
    maintainer_email = "housy17@mails.tsinghua.edu.cn",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = [
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "keras",
        "deepchem",
        "cairosvg",
        "bokeh",
        ]
)
