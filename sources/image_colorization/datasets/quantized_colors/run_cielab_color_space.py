#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from config import *
get_ipython().run_line_magic('run', '$current_dir/main.py main-compute-prior-prob')

