# -*- coding: utf-8 -*-
# CLI entry: run from repo root. Used in Render buildCommand.
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.fetch_pretrained_models import main

if __name__ == "__main__":
    raise SystemExit(main())
