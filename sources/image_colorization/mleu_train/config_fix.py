# config root dir, library path
import os, sys

root_dir      = os.path.abspath('../../..').replace("\\", "/")
source_dir    = os.path.join(root_dir, "sources").replace("\\", "/")
libraries_dir = os.path.join(root_dir, "libraries").replace("\\", "/")
include_dirs  = [source_dir, \
]
for lib in include_dirs:
    if lib not in sys.path: sys.path.insert(0, lib)
# np.set_printoptions(precision=2, suppress=True, formatter={'float': '{: 0.4f}'.format}, linewidth=1000)

# common info of project
data_dir       = os.path.join(root_dir, "data").replace("\\", "/")
dataset_dir    = os.path.join(data_dir, "datasets").replace("\\", "/")
exp_dir        = os.path.join(data_dir, "experiments").replace("\\", "/")

# path of module
current_dir       = os.path.dirname(os.path.abspath(__file__))
module_dir        = os.path.abspath(current_dir).replace("\\", "/")
relate_module_dir = os.path.relpath(module_dir, start=source_dir).replace("\\", "/")

def print_common_info(print=print):
    print("Common Config: ")
    for s_dir in ["root_dir", "source_dir", "libraries_dir", "data_dir", "dataset_dir", "exp_dir", "current_dir"]:
        print(f'+ {s_dir}: {globals()[s_dir]}')
    print()
# print_common_info