#!/usr/bin/env python
"""
Simple script to run the FX modeling pipeline.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import FXModelingPipeline

if __name__ == "__main__":
    pipeline = FXModelingPipeline("configs/config.yaml")
    pipeline.run()