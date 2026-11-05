"""Thin unified dispatcher for the local HorizonKV-EGate method."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from methods.horizonkv_egate.summarize import main
if __name__ == "__main__":
    sys.exit(main())
