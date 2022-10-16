import IPython

from describe import describe, load_semantic_map
from example_rms import office_t3
import rm_compiler


if __name__ == "__main__":
    semantic_map = load_semantic_map('../semantic_maps/t1.txt')
    rm = office_t3()

    rm_src = '(f(n)*e|ef)g'
    rm_compiler.compile(rm_src, frozenset('efgn'))

    # print(describe(semantic_map, rm))
