#!/usr/bin/env python
"""python scripts/preprocess.py -c configs/hausa_single.yaml"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hausafs2.data.preprocess import Preprocessor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True)
a = ap.parse_args()
Preprocessor(yaml.safe_load(open(a.config, encoding="utf-8"))).run()
