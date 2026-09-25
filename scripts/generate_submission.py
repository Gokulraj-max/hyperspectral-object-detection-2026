"""
Generate Submission CSV CLI:
python scripts/generate_submission.py --predictions outputs/predictions --output outputs/submissions/submission.csv
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from submission.generate_submission import main

if __name__ == "__main__":
    main()
