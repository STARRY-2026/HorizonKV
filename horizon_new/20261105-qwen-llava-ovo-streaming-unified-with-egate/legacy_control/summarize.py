#!/usr/bin/env python3
"""Summarize outputs through the backend that generated their manifest."""
import json,sys,subprocess
from pathlib import Path
if len(sys.argv)!=2:sys.exit('Usage: python summarize.py OUTPUT_DIR')
o=Path(sys.argv[1]);p=o/'manifest.json';p=p if p.exists() else o/'unified_manifest.json'
if not p.exists():sys.exit('No result manifest found')
m=json.loads(p.read_text());h=Path(__file__).resolve().parent
if m.get('model')=='qwen25vl':
 c=h/('qwen_full2495' if m.get('benchmark')=='streamingbench' and m.get('cohort')=='full' and m.get('questions')==2495 else 'qwen')
elif m.get('model')=='llavaov':c=h/'llava'
else:sys.exit('Unknown model')
sys.exit(subprocess.call([sys.executable,str(c/'summarize.py'),str(o)]))
