# config root dir, library path
import os, sys

current_dir   = os.path.dirname(os.path.abspath(__file__))
root_dir      = os.path.abspath('../../../../..').replace("\\", "/")
source_dir    = os.path.join(root_dir, "sources").replace("\\", "/")
if source_dir not in sys.path: sys.path.insert(0, source_dir)
# np.set_printoptions(precision=2, suppress=True, formatter={'float': '{: 0.4f}'.format}, linewidth=1000)

def print_init_info(print=print):
    print("Init Config: ")
    for s_dir in ["root_dir", "source_dir", "current_dir"]:
        print(f'+ {s_dir}: {globals()[s_dir]}')
    print()
# print_init_info