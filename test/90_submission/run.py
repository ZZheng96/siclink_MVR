"""Train the selected pipeline and create a clean final Kaggle archive."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.final import main

if __name__ == "__main__":
    main()
