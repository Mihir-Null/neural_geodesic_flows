"""
Inspect saved high-level parameters for trained RNGF/PRNGF models.
Prints the JSON contents for quick verification (e.g., signature, dims).
"""

import json
from pathlib import Path
from pprint import pprint


def main():
    model_dir = Path("data/models")
    files = [
        "minkowski_prngf_high_level_params.json",
        "minkowski_rngf_high_level_params.json",
        "ads2_prngf_high_level_params.json",
        "ads2_rngf_high_level_params.json",
        "schwarzschild_prngf_high_level_params.json",
        "schwarzschild_rngf_high_level_params.json",
    ]
    for name in files:
        path = model_dir / name
        if not path.exists():
            print(f"Missing {path}")
            continue
        with open(path) as f:
            print(path)
            pprint(json.load(f))
            print()


if __name__ == "__main__":
    main()
