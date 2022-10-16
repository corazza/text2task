import IPython

from describe import describe, load_semantic_map
from example_rms import office_t3
import rm_parser


if __name__ == "__main__":
    semantic_map = load_semantic_map('../semantic_maps/t1.txt')
    rm = office_t3()

    IPython.embed()

    parsed = rm_parser.parse('')

    print(describe(semantic_map, rm))
