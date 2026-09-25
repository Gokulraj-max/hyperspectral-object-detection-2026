"""
Inference Script CLI:
python scripts/predict.py --checkpoint checkpoints/best.pt --input data/test --output outputs/predictions --tta
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predict import main

if __name__ == "__main__":
    main()
