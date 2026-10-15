"""Symbol inventory and categorical feature vocabularies for the Hausa frontend.

All feature spaces are small closed sets; they are embedded separately in the
encoder (factored linguistic embeddings) rather than being flattened into one
giant phone vocabulary.
"""

PAD = "<pad>"

# Hausa consonants in standard (boko) orthography. Digraphs are single phonemes:
#   sh /ʃ/, ts /ts'/, fy /fʲ/, ky /kʲ/, gy /gʲ/, ƙy /k'ʲ/, kw /kʷ/, gw /gʷ/, ƙw /k'ʷ/.
# ɓ ɗ are implosives, ƙ is an ejective, ƴ is glottalised y, ʔ is the glottal stop (').
CONSONANTS = [
    "b", "ɓ", "c", "d", "ɗ", "f", "fy", "g", "gw", "gy", "h", "j",
    "k", "kw", "ky", "ƙ", "ƙw", "ƙy", "l", "m", "n", "r", "s", "sh",
    "t", "ts", "w", "y", "ƴ", "z", "ʔ",
]
VOWELS = ["a", "e", "i", "o", "u"]
DIPHTHONGS = ["ai", "au"]          # bimoraic nuclei
# Pause tokens (always present at word boundaries; duration may be 0 frames):
#   sp  = word boundary,  pm = minor phrase boundary (comma),  sil = major boundary (. ? !)
PAUSES = ["sp", "pm", "sil"]

PHONES = [PAD] + CONSONANTS + VOWELS + DIPHTHONGS + PAUSES
PHONE2ID = {p: i for i, p in enumerate(PHONES)}
N_PHONES = len(PHONES)
PAD_ID = 0

# --- categorical feature vocabularies ---------------------------------------
# Tone (only vowels / nuclei carry tone; everything else is NONE)
TONE_NONE, TONE_H, TONE_L, TONE_F, TONE_UNK = 0, 1, 2, 3, 4
TONE_NAMES = ["-", "H", "L", "F", "?"]
N_TONES = 5

# Length: vowel length / geminacy / diphthong bimoraicity
LEN_SHORT, LEN_LONG = 0, 1
N_LENGTHS = 2

# Syllable weight of the syllable a token belongs to
W_NONE, W_LIGHT, W_HEAVY, W_SUPER = 0, 1, 2, 3     # CV | CVV,CVC | CVVC,CVCC
WEIGHT_NAMES = ["-", "light", "heavy", "super"]
N_WEIGHTS = 4

# Role of a token inside its syllable
ROLE_ONSET, ROLE_NUCLEUS, ROLE_CODA, ROLE_NONE = 0, 1, 2, 3
N_ROLES = 4

# Boundary strength carried by pause tokens
B_NONE, B_WORD, B_MINOR, B_MAJOR = 0, 1, 2, 3
N_BOUNDARIES = 4

# Position of a unit inside its parent: none | initial | medial | final | single
POS_NONE, POS_INITIAL, POS_MEDIAL, POS_FINAL, POS_SINGLE = 0, 1, 2, 3, 4
N_POS = 5

# Token kind
KIND_C, KIND_V, KIND_P = 0, 1, 2

# Downstep counter vocabulary: 0..MAX_DOWNSTEP
MAX_DOWNSTEP = 7
N_DOWNSTEP = MAX_DOWNSTEP + 1

TONE_MARKS = {"\u0300": TONE_L, "\u0302": TONE_F, "\u0301": TONE_H}   # grave, circumflex, acute
MACRON = "\u0304"
