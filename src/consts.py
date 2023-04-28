TRAIN_LR = 1e-3
PAD_SIZE = 256

SOURCES = ['organic1',
           'organic2',
           #   'organic3',
           'organic_words',
           'organic_interesting',
           'organic_asdf'
           ]

SENTENCE_CAP = 40

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
