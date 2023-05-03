TRAIN_LR = 1e-3
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
DEMORGANS_P = 0.3

ADD_AVOIDANCE_P = 0.35
ADD_CONCAT_P = 0.35
ADD_DISJUNCT_P = 0.2
ADD_CONJUNCT_P = 0.1
ADD_P = 1
PARAP_P = 0.99
assert abs(ADD_AVOIDANCE_P + ADD_CONCAT_P +
           ADD_DISJUNCT_P + ADD_CONJUNCT_P - 1) < 1e-9

AUGMENT_PREFER_BEFORE = 0.4
DESC_LENGTH_LIMIT = 50

REWRITE_VALIDATION_EMPTY_PROB = 0.3

AUGMENT_CHAR_LIST = ['P', 'Q', 'X', 'Y', 'Z', 'W']

POSNEG_VALIDATION = True
VALIDATE_RAW = False
VALIDATE_AUGMENTED = False
VALIDATE_AB_AUGMENTS = False

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
