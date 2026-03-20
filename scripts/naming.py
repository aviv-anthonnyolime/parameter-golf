"""
Docker-style adjective-animal name generator for training runs.

Usage:
    from scripts.naming import generate_run_name
    name = generate_run_name()           # random: "brave-falcon"
    name = generate_run_name(seed=42)    # deterministic: same seed → same name
"""

import hashlib
import random

ADJECTIVES = [
    "agile", "bold", "brave", "bright", "calm", "clever", "cool", "cosmic",
    "crisp", "dark", "deep", "eager", "epic", "fair", "fast", "fierce",
    "fleet", "frosty", "gentle", "grand", "happy", "hardy", "keen", "kind",
    "lively", "lucky", "merry", "mighty", "noble", "plucky", "proud", "quick",
    "quiet", "rapid", "rustic", "sharp", "shiny", "silent", "sleek", "smooth",
    "snappy", "solar", "solid", "spicy", "stark", "steady", "stoic", "strong",
    "super", "swift", "tidy", "tough", "vivid", "warm", "wild", "wise",
    "witty", "zany", "zen", "zippy",
]

ANIMALS = [
    "alpaca", "badger", "bear", "bison", "cobra", "condor", "crane", "crow",
    "deer", "dingo", "eagle", "egret", "elk", "falcon", "ferret", "finch",
    "fox", "gecko", "goose", "gorilla", "hawk", "heron", "horse", "husky",
    "ibis", "iguana", "impala", "jackal", "jaguar", "jay", "kite", "koala",
    "lark", "lemur", "lion", "llama", "lynx", "macaw", "mink", "moose",
    "newt", "okapi", "orca", "osprey", "otter", "owl", "panda", "parrot",
    "puma", "quail", "raven", "robin", "salmon", "seal", "shark", "sloth",
    "snake", "squid", "stork", "swan", "tiger", "toucan", "trout", "viper",
    "whale", "wolf", "wren", "yak", "zebra", "zorilla",
]


def generate_run_name(seed=None):
    """Generate a docker-style adjective-animal name."""
    rng = random.Random(seed)
    adj = rng.choice(ADJECTIVES)
    animal = rng.choice(ANIMALS)
    return f"{adj}-{animal}"


def generate_run_name_from_string(s: str):
    """Deterministic name from any string (e.g. run_id UUID)."""
    h = int(hashlib.sha256(s.encode()).hexdigest(), 16)
    return generate_run_name(seed=h)


def params_tag(num_layers, num_unique_layers, model_dim, mlp_mult):
    """Short param summary string: '10L-1U-512d-2mlp'."""
    return f"{num_layers}L-{num_unique_layers}U-{model_dim}d-{mlp_mult}mlp"


def make_log_filename(date_str, time_str, param_tag, docker_name):
    """Build the log filename: 2026-03-20_143052_10L-1U-512d-2mlp_brave-falcon.txt"""
    return f"{date_str}_{time_str}_{param_tag}_{docker_name}.txt"


if __name__ == "__main__":
    # Demo
    for i in range(10):
        print(generate_run_name(seed=i))
