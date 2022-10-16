import IPython

from describe import describe, load_semantic_map
from example_rms import office_t3
import compiler_interface


if __name__ == "__main__":
    semantic_map = load_semantic_map('../semantic_maps/t1.txt')
    rm = office_t3()

    rm_src = '(f(n)*e|ef)g'
    compiler_interface.compile(rm_src, frozenset('efgn'))

    # print(describe(semantic_map, rm))
