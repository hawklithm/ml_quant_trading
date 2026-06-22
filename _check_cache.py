#!/usr/bin/env python3
import os, glob, pickle, pandas as pd

path = os.path.expanduser("~/.cache/hermes-quant/crypto/")
print("=== Crypto cache dir ===")
for f in sorted(glob.glob(path + "*")):
    sz = os.path.getsize(f)
    print(f"  {os.path.basename(f)}: {sz} bytes")

print("=== Looking for BTC/ETH/SOL in main cache ===")
for f in sorted(glob.glob(os.path.expanduser("~/.cache/hermes-quant/data_*.pkl"))):
    bn = os.path.basename(f)
    if "BTC" in bn or "ETH" in bn or "SOL" in bn:
        print(f"  {bn}: {os.path.getsize(f)} bytes")
