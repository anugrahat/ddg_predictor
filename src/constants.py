"""Global constants used across the project."""

AA_ORDER = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X",
]
AA_TO_ID = {aa: i for i, aa in enumerate(AA_ORDER)}

EDGE_INTRA_SPATIAL = 0
EDGE_SEQUENTIAL = 1
EDGE_INTER_CHAIN = 2
