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

# --- GPU memory estimate for DDP training ---
# In DDP each GPU holds the full model + gradients + optimizer states.
#
# Classify params: "matrix" → Muon (1 momentum), everything else → Adam (2 moments).
# Activation memory is the dominant cost: traced per-layer through the actual
# architecture (Block → Attention + MLP) with flash-attention SDPA.

matrix_params = 0
scalar_params = 0
for name, p in model.named_parameters():
    if p.ndim >= 2:
        matrix_params += p.numel()
    else:
        scalar_params += p.numel()

# --- static memory (same regardless of batch size) ---
model_bf16    = n_params * 2          # weights in BF16
grad_fp32     = n_params * 4          # gradients in FP32
muon_states   = matrix_params * 4     # 1 momentum buffer (FP32)
adam_states   = scalar_params * 8     # m1 + m2 (FP32 each)
ddp_buckets   = n_params * 2          # DDP gradient comm buckets (BF16)

static_bytes  = model_bf16 + grad_fp32 + muon_states + adam_states + ddp_buckets

# --- activation memory per layer (no gradient checkpointing) ---
# Traced through Block.forward → Attention (SDPA flash) + MLP (ReLU²):
#
# Attention saved tensors (BF16, 2 bytes each):
#   block input x (residual)       : B × S × D
#   attn_norm input                : B × S × D
#   normalized n (shared by QKV)   : B × S × D
#   Q pre-rms_norm                 : B × S × D
#   K pre-rms_norm                 : B × S × kv_dim
#   Q post-rope+gain (SDPA input)  : B × S × D
#   K post-rope (SDPA input)       : B × S × kv_dim
#   V (SDPA input)                 : B × S × kv_dim
#   SDPA output                    : B × S × D
#   proj input (contiguous)        : B × S × D
#   x post-attn (MLP norm input)   : B × S × D
#
# MLP saved tensors (BF16):
#   mlp_norm output (fc input)     : B × S × D
#   relu mask (1 bit per element)  : B × S × hidden / 8
#   relu output (saved by square)  : B × S × hidden
#   squared (proj input)           : B × S × hidden
#
# Totals per layer:
#   D-sized tensors  : ~9  × B × S × D × 2
#   KV-sized tensors : ~3  × B × S × kv_dim × 2
#   hidden-sized     : ~2  × B × S × hidden × 2  (+ small relu mask)

batch_tokens  = getattr(args, "train_batch_tokens", 524_288)
seq_len       = getattr(args, "train_seq_len", 1024)
model_dim     = getattr(args, "model_dim", gpt_kwargs.get("dim", 512))
num_layers    = getattr(args, "num_layers", gpt_kwargs.get("num_layers", 9))
num_heads     = getattr(args, "num_heads", gpt_kwargs.get("num_heads", 8))
num_kv_heads  = getattr(args, "num_kv_heads", gpt_kwargs.get("num_kv_heads", 4))
mlp_mult      = getattr(args, "mlp_mult", gpt_kwargs.get("mlp_mult", 2))
n_gpus        = int(os.environ.get("NPROC", 4))
gpu_budget_gb = float(os.environ.get("GPU_MEM_GB", 24.0))

kv_dim = model_dim * num_kv_heads // num_heads
hidden = model_dim * mlp_mult
n_tokens_per_gpu = batch_tokens // max(n_gpus, 1)
micro_batch = n_tokens_per_gpu // seq_len

# Per-layer activation bytes (BF16 = 2 bytes per element)
d_tensors   = 9 * n_tokens_per_gpu * model_dim * 2
kv_tensors  = 3 * n_tokens_per_gpu * kv_dim * 2
mlp_tensors = 2 * n_tokens_per_gpu * hidden * 2
relu_mask   = n_tokens_per_gpu * hidden // 8        # 1-bit mask
act_per_layer = d_tensors + kv_tensors + mlp_tensors + relu_mask

act_bytes = act_per_layer * num_layers

# CUDA context + fragmentation overhead
cuda_overhead = int(1.0 * 1024**3)  # ~1 GB typical

total_bytes = static_bytes + act_bytes + cuda_overhead
GB = 1024 ** 3
fits_ddp = (total_bytes / GB) <= gpu_budget_gb

# --- suggest reduced batch if OOM ---
if not fits_ddp:
    # binary search for max batch_tokens that fits
    max_batch = batch_tokens
    for candidate in [batch_tokens // 2, batch_tokens // 4, batch_tokens // 8]:
        tpg = candidate // max(n_gpus, 1)
        cand_act = (9*tpg*model_dim*2 + 3*tpg*kv_dim*2 + 2*tpg*hidden*2 + tpg*hidden//8) * num_layers
        if (static_bytes + cand_act + cuda_overhead) / GB <= gpu_budget_gb:
            max_batch = candidate
            break
    else:
        max_batch = 0

# --- report ---
print(f"\n{'='*60}")
print(f"  {script_path}")
print(f"{'='*60}")
for k, v in gpt_kwargs.items():
    print(f"  {k:<24} {v}")
print(f"{'─'*60}")
print(f"  total params           {n_params/1e6:.3f}M")
print(f"    matrix (Muon)        {matrix_params/1e6:.3f}M")
print(f"    scalar/embed (Adam)  {scalar_params/1e6:.3f}M")
print(f"{'─'*60}")
print(f"  int8 raw               {raw_bytes/1024**2:.2f} MB")
print(f"  int8 + zlib (level 9)  {zlib_bytes/1024**2:.2f} MB  ← artifact size")
print(f"  16MB budget left       {(limit - zlib_bytes)/1024**2:.2f} MB")
print(f"  {'✓ fits' if zlib_bytes < limit else '✗ OVER 16MB'}")
print(f"{'─'*60}")
print(f"  GPU memory / device (DDP, {n_gpus}×{gpu_budget_gb:.0f}GB)")
print(f"  ┌ static")
print(f"  │  model BF16           {model_bf16/GB:.2f} GB")
print(f"  │  gradients FP32       {grad_fp32/GB:.2f} GB")
print(f"  │  Muon states          {muon_states/GB:.2f} GB")
print(f"  │  Adam states          {adam_states/GB:.2f} GB")
print(f"  │  DDP buckets          {ddp_buckets/GB:.2f} GB")
print(f"  │  subtotal             {static_bytes/GB:.2f} GB")
print(f"  ├ activations (TRAIN_BATCH_TOKENS={batch_tokens}, {micro_batch}seq×{seq_len}len)")
print(f"  │  per layer            {act_per_layer/GB:.3f} GB × {num_layers} layers")
print(f"  │    D-sized  (9×B×S×D)          {d_tensors/GB:.3f} GB")
print(f"  │    KV-sized (3×B×S×kv)         {kv_tensors/GB:.3f} GB")
print(f"  │    MLP-sized(2×B×S×{hidden})    {mlp_tensors/GB:.3f} GB")
print(f"  │  subtotal             {act_bytes/GB:.2f} GB")
print(f"  ├ CUDA overhead         {cuda_overhead/GB:.2f} GB")
print(f"  └ TOTAL                 {total_bytes/GB:.2f} GB  / {gpu_budget_gb:.0f} GB budget")
if fits_ddp:
    print(f"  ✓ fits on {n_gpus}×{gpu_budget_gb:.0f}GB  (headroom: {(gpu_budget_gb - total_bytes/GB):.2f} GB)")
else:
    print(f"  ✗ OOM on {n_gpus}×{gpu_budget_gb:.0f}GB  (over by {(total_bytes/GB - gpu_budget_gb):.2f} GB)")
    if max_batch > 0:
        tpg = max_batch // max(n_gpus, 1)
        cand_act = (9*tpg*model_dim*2 + 3*tpg*kv_dim*2 + 2*tpg*hidden*2 + tpg*hidden//8) * num_layers
        cand_total = (static_bytes + cand_act + cuda_overhead) / GB
        print(f"  → try TRAIN_BATCH_TOKENS={max_batch} ({tpg//seq_len}seq/gpu) → ~{cand_total:.2f} GB")
    else:
        print(f"  → model too large even at batch=1, reduce MODEL_DIM or NUM_LAYERS")
print(f"{'='*60}\n")
