from describe import describe, load_semantic_map
from reward_machines.example_rms import office_t3


if __name__ == "__main__":
    semantic_map = load_semantic_map('../semantic_maps/t1.txt')
    rm = office_t3()
    print(describe(semantic_map, rm))
