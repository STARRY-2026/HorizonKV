"""Unified H2SVC summary entry."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.h2svc.summarize import main

if __name__ == '__main__':
    main()
