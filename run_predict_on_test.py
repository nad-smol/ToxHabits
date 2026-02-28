from __future__ import annotations

import argparse
import os
import sys
import subprocess

SUPPLEMENTARY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SUPPLEMENTARY_DIR))

parser = argparse.ArgumentParser(description="Model run for test sample")
parser.add_argument("--task", required=True, choices=["toxner", "toxuse"], help="Task: toxner or toxuse")
parser.add_argument("--checkpoint", required=True, help="Model (.pt) directory")
parser.add_argument("--test_dir", default=None, help="Test sample directory")
parser.add_argument("--out_dir", default=None, help="Output data directory")
args = parser.parse_args()

checkpoint = args.checkpoint
if not os.path.isabs(checkpoint):
    checkpoint = os.path.normpath(os.path.join(PROJECT_ROOT, checkpoint))
if not os.path.isfile(checkpoint):
    print(f"Чекпоинт не найден: {checkpoint}")
    sys.exit(1)

ckpt_basename = os.path.basename(checkpoint).lower()
if ckpt_basename.startswith("a1_beto"):
    predict_script = os.path.join(PROJECT_ROOT, "predict_A1_BETO_improved.py")
else:
    predict_script = os.path.join(PROJECT_ROOT, "predict_A2_bsc_improved.py")

if not os.path.isfile(predict_script):
    print(f"Python script not found")
    sys.exit(1)

cmd = [
    sys.executable,
    predict_script,
    "--task", args.task,
    "--checkpoint", checkpoint,
]
if args.test_dir:
    test_dir = args.test_dir if os.path.isabs(args.test_dir) else os.path.join(PROJECT_ROOT, args.test_dir)
    cmd += ["--test_dir", os.path.normpath(test_dir)]
if args.out_dir:
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(PROJECT_ROOT, args.out_dir)
    cmd += ["--out_dir", os.path.normpath(out_dir)]

print("Run: ", " ".join(cmd))
sys.exit(subprocess.run(cmd, cwd=PROJECT_ROOT).returncode)


