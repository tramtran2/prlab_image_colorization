import os, sys, json5
from prlab.utils.config import *

module_dir   = os.path.dirname(os.path.abspath(__file__))

############################################
def load_config(parent_globals, print = print):
    global global_cfg, config_file, root_dir, config_dir
    
    root_dir      = parent_globals["root_dir"]
    config_file   = f'{root_dir}/sources/config.json'
    config_dir    = os.path.dirname(os.path.abspath(config_file))
    global_cfg = load_json_config(config_file, parent_globals, globals(), remove_meta = True)
    for lib in global_cfg["include_dirs"]:
        if lib not in sys.path: sys.path.insert(0, lib)
    # for
    parent_globals.update(global_cfg=global_cfg,config_file=config_file,config_dir=config_dir)
    print_load_info(print = print)
# load_config
    
def print_load_info(print=print):
    global global_cfg, config_file
    
    print("Load Module Config: ")
    for s_dir in ["root_dir", "source_dir", "libraries_dir", "data_dir", "dataset_dir", "exp_dir", "module_dir", "current_dir"]:
        print(f'+ {s_dir}: {global_cfg[s_dir]}')
    print(f'+ global_cfg: \t{list(global_cfg.keys())}')
    print(f'+ config_file: \t{config_file}')
    print()
# print_load_info
############################################