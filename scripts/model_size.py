"""
Instantiates the real model and runs the actual int8+zlib compression.
No training, no data, no GPU required.

Usage (from repo root):
    python scripts/model_size.py experiments/phase1_UT/train_gpt_ut.py
    python scripts/model_size.py train_gpt.py
    NUM_LAYERS=10 NUM_UNIQUE_LAYERS=2 MODEL_DIM=640 python scripts/model_size.py experiments/phase1_UT/train_gpt_ut.py

Automatically matches Hyperparameters fields to GPT.__init__ signature,
so it works with any version of the script regardless of added/removed args.
"""

import importlib.util, inspect, io, os, sys, zlib
import torch

if len(sys.argv) < 2:
    print("Usage: python scripts/model_size.py <path/to/train_gpt_*.py>")
    sys.exit(1)

script_path = sys.argv[1]
if not os.path.isfile(script_path):
    print(f"File not found: {script_path}")
    sys.exit(1)

# --- dynamic import ---
spec = importlib.util.spec_from_file_location("train", script_path)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

Hyperparameters          = train.Hyperparameters
GPT                      = train.GPT
quantize_state_dict_int8 = train.quantize_state_dict_int8

# --- build kwargs: only pass what GPT.__init__ actually accepts ---
args = Hyperparameters()
gpt_sig = inspect.signature(GPT.__init__).parameters
hp_fields = {k: v for k, v in vars(args.__class__).items() if not k.startswith("_")}
gpt_kwargs = {k: hp_fields[k] for k in gpt_sig if k in hp_fields}

model = GPT(**gpt_kwargs)
n_params = sum(p.numel() for p in model.parameters())

# --- real quantization + compression ---
quant_obj, _ = quantize_state_dict_int8(model.state_dict())
buf = io.BytesIO()
torch.save(quant_obj, buf)
raw_bytes  = len(buf.getvalue())
zlib_bytes = len(zlib.compress(buf.getvalue(), level=9))
limit      = 16 * 1024**2

# --- report ---
unique = getattr(args, "num_unique_layers", getattr(args, "num_layers", "?"))
print(f"\n{'='*52}")
print(f"  {script_path}")
print(f"{'='*52}")
for k, v in gpt_kwargs.items():
    print(f"  {k:<24} {v}")
print(f"{'─'*52}")
print(f"  total params           {n_params/1e6:.3f}M")
print(f"{'─'*52}")
print(f"  int8 raw               {raw_bytes/1024**2:.2f} MB")
print(f"  int8 + zlib (level 9)  {zlib_bytes/1024**2:.2f} MB  ← artifact size")
print(f"  16MB budget left       {(limit - zlib_bytes)/1024**2:.2f} MB")
print(f"  {'✓ fits' if zlib_bytes < limit else '✗ OVER 16MB'}")
print(f"{'='*52}\n")
