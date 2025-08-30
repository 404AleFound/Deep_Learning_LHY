# utils.py --
# Le Jiang
# 2025/8/30

import os

def makedir_(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o777)

def makedirs_(paths):
    for path in paths:
        makedir_(path)
