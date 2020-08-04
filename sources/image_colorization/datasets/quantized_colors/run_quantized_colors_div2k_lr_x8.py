#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function
from IPython import get_ipython

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from config import *

# %run $current_dir/main.py main-compute-prior-prob
cmd_args  = f' --db_root "{dataset_dir}/div2k/outputs"'
cmd_args += f' --db_file "{dataset_dir}/div2k/outputs/preprocessing/div2k_lr_x8.hdf5"'
cmd_args += f' --db_name images'

cmd_args += f' --column_image image'
cmd_args += f' --column_type type'

cmd_args += f' --process_types train'
# cmd_args += f' --process_types valid'

cmd_args += f' --pts_in_hull_path "{data_dir}/colorization_richard_zhang/pts_in_hull.npy"'
cmd_args += f' --export_prior_prob_path "{dataset_dir}/div2k/outputs/preprocessing/prior_prob_train_div2k.npy"'

cmd_args += f' --is_resize False'
cmd_args += f' --width 112'
cmd_args += f' --height 112'

cmd_args += f' --do_plot True'
cmd_args += f' --verbose True'

get_ipython().run_line_magic('run', '-i $current_dir/main.py main-compute-prior-prob $cmd_args')

