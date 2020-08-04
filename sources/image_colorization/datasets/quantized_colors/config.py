import os, sys, json5

## TO-DO
current_dir   = os.path.dirname(os.path.abspath(__file__))
root_dir      = os.path.abspath(current_dir + '/../../../..').replace("\\", "/")
config_file   = f'{root_dir}/sources/config.json'

############################################
config_dir    = os.path.dirname(os.path.abspath(config_file))
if config_dir not in sys.path: sys.path.insert(0, config_dir)
from prlab.utils.config import *
from global_utils import *
from global_common import *

print(f'Config Info: ')
print(f'+ current_dir: \t{current_dir}')
print(f'+ root_dir: \t{root_dir}')
print(f'+ config_file: \t{config_file}')
print()

global_cfg = load_json_config(config_file, globals(), locals(), remove_meta = True)
for lib in global_cfg["include_dirs"]:
    if lib not in sys.path: sys.path.insert(0, lib)
# for
############################################