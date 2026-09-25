"""
Training Script CLI:
python scripts/train.py --config configs/final.yaml
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train import main

if __name__ == "__main__":
    main()
