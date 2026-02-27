"""
Standalone Training Script — India Paddy Yield (Real Data)
Run from project root:
    python3 train_standalone.py
"""

import sys
import os

# Add project to path so 'src' package resolves
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_pipeline

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train India Paddy Yield model (real data)')
    parser.add_argument('--rebuild', action='store_true',
                        help='Force rebuild of clean dataset from raw CSV')
    args = parser.parse_args()

    results = train_pipeline(force_rebuild=args.rebuild)
    print("\nDone! Model and reports saved.")
