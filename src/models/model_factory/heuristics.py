
# These support Regex
CLASSIFIER_KEYS = [
    ".*classifier",
]

POOLER_KEYS = [
    ".*pooler",
]

EXCLUDED_KEYS = [
    ".*bias",
]

# These do not support Regex
MAPPING_KEYS = {
    "classifier.weight": "classifier.linear.weight"
}