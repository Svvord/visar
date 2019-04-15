#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: S. Hou
# Mail: housy17@mails.tsinghua.edu.cn
# Created Time:  2019\04\14 22:30
#############################################


from setuptools import setup, find_packages

setup(
    name = "visar",
    version = "0.1.3",
    keywords = ("pip", "pathtool","timetool", "magetool", "mage"),
    description = "xxx",
    long_description = "xxx",
    license = "MIT Licence",

    url = "https://github.com/Svvord/xxx",
    author = "S. Hou",
    author_email = "housy17@mails.tsinghua.edu.cn",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = []
)