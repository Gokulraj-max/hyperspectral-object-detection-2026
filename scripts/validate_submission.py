"""
Validate Submission CSV CLI:
python scripts/validate_submission.py --input outputs/submissions/submission.csv
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from submission.validate_submission import main

if __name__ == "__main__":
    main()
