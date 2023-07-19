# ========== COMMON ==============
SEED: int = 42

# ========== DATA AGUMENTATION ==============
SOURCES: list[str] = [
    'organic1_loose',  # looser validation
    'organic1_nonloose',  # moved from 1/2 after loose checking
    'organic_nonverified',
    'organic_words',
]

SOURCES_SINGLE_LINE: list[str] = [
    '1',
]


COMBINATIONS_CUTOFF = 100
SENTENCE_CAP = 3

TAKE_DEMORGANS = 1
TAKE_OTHERS = 50
DEMORGANS_P = 0.001

ADD_CONCAT_P = 4
ADD_AVOIDANCE_P = 3
ADD_AVOIDANCE_BOTH_P = 1
ADD_CONCAT_AVOID_P = 3
ADD_CONCAT_CONCAT_P = 3
ADD_DISJUNCT_P = 2
ADD_CONJUNCT_P = 1
ADD_DISJUNCT_CONCAT_P = 1
ADD_CONCAT_DISJUNCT_P = 1

ADD_P = 20

ADD_TOTAL = ADD_AVOIDANCE_P + ADD_AVOIDANCE_BOTH_P + ADD_CONCAT_P + ADD_CONCAT_AVOID_P + \
    ADD_DISJUNCT_P + ADD_CONJUNCT_P + ADD_CONCAT_CONCAT_P + ADD_DISJUNCT_CONCAT_P + \
    ADD_CONCAT_DISJUNCT_P

REWRITE_VALIDATION_EMPTY_PROB = 0.3

AUGMENT_CHAR_LIST = ['P', 'Q', 'X', 'Y', 'Z', 'W']
CANT_APPEAR_IN_BOTH = ['first', 'second', 'finally', ':', 'either', 'lastly']
CANT_APPEAR_IN_SINGLE = ['.', ':', 'never',
                         "don't", 'do not', 'allowed', 'whatever']

POSNEG_VALIDATION = True
VALIDATE_AUGMENTED = False

TIMES_CHAR_LIST = ['N', 'M', 'O']
DEFAULT_TIMES = 2
HIGHEST_TIMES = 5
TIMES_MAP = {
    1: ["one time", "1 time", "once"],
    2: ["two times", "2 times", "twice"],
    3: ["three times", "3 times", "thrice"],
    4: ["four times", "4 times"],
    5: ["five times", "5 times"]
}
NUM_MAP = {
    # 1: ["one", "1"],
    2: ["two", "2"],
    3: ["three", "3"],
    4: ["four", "4"],
    5: ["five", "5"]
}

DEFAULT_TERMS_PATH = 'datasets/txt2task/terms2.txt'
PAD_SIZE = 100


# ========== TRANSFORMER ==============
TRAIN_LR = 0.5e-3
EVAL_STEPS_WARMUP = 20
EVAL_STEPS_OVERWRITE = 10


# ========== MODEL USAGE ==============
# DEFAULT_USE_MODEL_PATH: str = '/mnt/e/work_dirs/text2task_distilgpt2/'
DEFAULT_USE_MODEL_PATH: str = '/mnt/e/work_dirs/text2task_gpt2/'
MODEL_TEST_TEMPERATURE: float = 1.2
# go to the store. avoid traps. then, go to the town
MODEL_NUM_RETURN_SEQUENCES: int = 50
SEMANTIC_SIMILARITY_NUM_CLUSTERS: int = 3
SEMANTIC_SIMILARITY_MAX_LENGTH: int = 50
SEMANTIC_SIMILARITY_NUM_SAMPLES: int = 500
SEMANTIC_SIMILARITY_SAMPLES_REDUNDANCY: int = 10

# ========== REINFORCEMENT LEARNING ==============
DEFAULT_MAP_PATH = 'preprocessed_datasets/txt2task/map_test.txt'
DEFAULT_AGENT: str = 'qrm'
DEFAULT_STEPS: int = int(0.5e+05)
# DEFAULT_STEPS: int = 16000
PER_EPISODE_STEPS: int = int(1e+03)
DEFAULT_Q_INIT: float = 0.2

N_DEMO_EPISODES: int = 10
N_DEMO_MAX_STEPS: int = 1000
SHOW_COMPLETION: bool = True

NETURAL_COLOR = (0.5, 0.5, 0.5)

# =================

# go to the house. then, find a coin in the town
# go to the town. then, find green cans in the forest. but avoid mines -> CAN, but no green

# find zero cans -> CAN
# ((.)* > can > (.)*)~

# find no cans -> CAN
# ((.)* > can > (.)*)~&((.)* > can > (.)*)~

# don't find cans -> CAN
# ((.)* > can > (.)*)~&((.)* > can > (.)*)~

# find a positive number of cans -> CAN
# ((.)* > can){#some}

# find several cans
# ((.)* > can){#some}

# find a garden in the town, but avoid traps 5
# ((.)* > garden&town)&((.)* > trap > (.)*)~

# patrol the hospital and the building three times 6

# go to the store, but avoid mines and traps. then, find a coin in the town


###

# patrol the field and the town
# ((.)* > field > (.)* > town > (.)* > field){#some}

# find a key
# (.)* > key
