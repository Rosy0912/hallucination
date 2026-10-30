"""Offline recomputation; no GPU or model required."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.score_binary import main
if __name__ == '__main__': main()
