"""Wrapper: apply masked attention patch, then run LoRAEdit training."""
import sys
import os

LORAEDIT_DIR = '/home/users/ntu/song0304/code/LoRAEdit'
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup paths before any imports
sys.path.insert(0, LORAEDIT_DIR)
sys.path.insert(0, EXPERIMENT_DIR)
os.chdir(LORAEDIT_DIR)

# Step 1: Load mask and apply attention patch
from patch import apply_patch, load_mask

DATA_DIR = '/scratch/users/ntu/song0304/loraedit_exp/processed_data/V7_hair'
load_mask(DATA_DIR)
apply_patch()

# Step 2: Execute the original train.py in the same namespace
# This preserves __name__ == '__main__' needed by argparse/deepspeed
exec(compile(
    open(os.path.join(LORAEDIT_DIR, 'train.py')).read(),
    os.path.join(LORAEDIT_DIR, 'train.py'),
    'exec'
))
