TRAIN_LR = 1e-3
EVAL_STEPS_WARMUP = 20
EVAL_STEPS_OVERWRITE = 10
SEED = 42
PAD_SIZE = 256

SOURCES = [
    'organic1_loose',  # looser validation
    'organic1_nonloose',  # moved from 1/2 after loose checking
    'organic_words',
]

SENTENCE_CAP = 10

COMBINATIONS_CUTOFF = 100

TAKE_DEMORGANS = 1
TAKE_OTHERS = 50
DEMORGANS_P = 0.001

ADD_CONCAT_P = 4
ADD_AVOIDANCE_P = 3
ADD_CONCAT_AVOID_P = 3
ADD_CONCAT_CONCAT_P = 3
ADD_DISJUNCT_P = 2
ADD_CONJUNCT_P = 1
ADD_DISJUNCT_CONCAT_P = 1
ADD_CONCAT_DISJUNCT_P = 1

ADD_P = 15

ADD_TOTAL = ADD_AVOIDANCE_P + ADD_CONCAT_P + ADD_CONCAT_AVOID_P + \
    ADD_DISJUNCT_P + ADD_CONJUNCT_P + ADD_CONCAT_CONCAT_P + ADD_DISJUNCT_CONCAT_P + \
    ADD_CONCAT_DISJUNCT_P

# PARAP_P = 0.99
# assert abs(ADD_AVOIDANCE_P + ADD_CONCAT_P + ADD_CONCAT_AVOID_P +
#            ADD_DISJUNCT_P + ADD_CONJUNCT_P + ADD_CONCAT_CONCAT_P + ADD_DISJUNCT_CONCAT_P +
#            ADD_CONCAT_DISJUNCT_P - 1) < 1e-9

AUGMENT_PREFER_BEFORE = 0.4
DESC_LENGTH_LIMIT = 100

REWRITE_VALIDATION_EMPTY_PROB = 0.3

AUGMENT_CHAR_LIST = ['P', 'Q', 'X', 'Y', 'Z', 'W']
CANT_APPEAR_IN_BOTH = ['first', 'second', 'finally', ':', 'either']
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
DEFAULT_MAP_PATH = 'preprocessed_datasets/txt2task/map_test.txt'

DEFAULT_AGENT: str = 'qrm'
DEFAULT_STEPS: int = int(1e+06)
DEFAULT_Q_INIT: float = 0.2

# DEFAULT_USE_MODEL_PATH: str = '/mnt/e/work_dirs/text2task_distilgpt2/'
DEFAULT_USE_MODEL_PATH: str = '/mnt/e/work_dirs/text2task_gpt2/'

MODEL_TEST_TEMPERATURE: float = 1.2
MODEL_NUM_RETURN_SEQUENCES: int = 10
SEMANTIC_SIMILARITY_NUM_CLUSTERS: int = 3
SEMANTIC_SIMILARITY_MAX_LENGTH: int = 50
SEMANTIC_SIMILARITY_NUM_SAMPLES: int = 500
SEMANTIC_SIMILARITY_SAMPLES_REDUNDANCY: int = 10
SEMANTIC_SIMILARITY_EMPTY_PROB: float = 0.7


# go to the house. then, find a coin in the town
# go to the town. then, find green cans in the forest. but avoid mines -> DIFFICULT

# find a positive number of cans -> CURRENTLY CAN'T
# find several cans -> CAN do
