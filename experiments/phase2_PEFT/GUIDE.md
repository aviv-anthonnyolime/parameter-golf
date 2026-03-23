# Phase 2 PEFT — Implementation Guide

## What you need to add

The script already has **TTT LoRA** (test-time training — re-initialized per eval batch).
You need a **different thing**: persistent per-step LoRA adapters trained alongside the model.

## Architecture recap

```
GPT.forward():
  x = tok_emb(input_ids)
  for i in range(num_layers):                          # 10 logical layers
      x += depth_embed[i]                              # already differentiates steps
      x = blocks[i % num_unique_layers](x, x0, ...)   # only 5 unique blocks
```

The 5 shared blocks are reused 2x each (encoder + decoder). Depth embeddings help,
but they only add a bias — LoRA adds a per-step *transformation*.

## Where to add per-step LoRA

### 1. Create a `PerStepLoRA` module (~30 lines)

```python
class PerStepLoRA(nn.Module):
    """One LoRA adapter per logical layer position (not per unique block)."""
    def __init__(self, num_steps, in_features, out_features, rank):
        super().__init__()
        # num_steps = num_layers (e.g. 10 for 10L)
        # Each step gets its own A and B matrices
        self.A = nn.ParameterList([
            nn.Parameter(torch.randn(rank, in_features) * 0.01)
            for _ in range(num_steps)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, rank))  # B=0 init!
            for _ in range(num_steps)
        ])

    def forward(self, x, step_idx):
        # x: (bsz, seq, in_features)
        # Returns delta: (bsz, seq, out_features)
        return (x @ self.A[step_idx].T) @ self.B[step_idx].T
```

**Key: B initialized to zeros** so all adapters start identical (output = 0), then diverge.

### 2. Add LoRA modules to GPT.__init__ (around line 845)

After creating `self.blocks`, add:

```python
# Per-step LoRA adapters (one per logical layer, applied to Q and V projections)
total_layers = self.num_encoder_layers + self.num_decoder_layers
if peft_rank > 0:
    dim = model_dim
    kv_dim = (model_dim // num_heads) * num_kv_heads
    self.q_lora = PerStepLoRA(total_layers, dim, dim, peft_rank)
    self.v_lora = PerStepLoRA(total_layers, dim, kv_dim, peft_rank)
    # Optional: MLP adapter too
    # self.mlp_lora = PerStepLoRA(total_layers, dim, dim, peft_rank)
else:
    self.q_lora = None
    self.v_lora = None
```

### 3. Update GPT.forward() (around line 880)

Replace the current LoRA calls (which pass TTT lora) with per-step ones:

```python
for i in range(self.num_encoder_layers):
    x = x + self.depth_embed.weight[i].to(dtype=x.dtype)[None, None, :]
    # Per-step LoRA: create delta functions for this layer position
    qd_fn = (lambda n, idx=i: self.q_lora(n, idx)) if self.q_lora else None
    vd_fn = (lambda n, idx=i: self.v_lora(n, idx)) if self.v_lora else None
    x = self.blocks[i % self.num_unique_layers](x, x0, qd_fn, vd_fn)
    skips.append(x)
```

Same pattern for the decoder loop. **Watch the `idx=i` default arg** — without it,
Python closures capture by reference and all lambdas would use the last `i`.

### 4. Add LoRA params to an optimizer (around line 1400)

LoRA matrices are 2D → they could go into Muon or Adam. Start with Adam:

```python
if base_model.q_lora is not None:
    lora_params = list(base_model.q_lora.parameters()) + \
                  list(base_model.v_lora.parameters())
    optimizer_lora = torch.optim.Adam(
        [{"params": lora_params, "lr": args.matrix_lr, "base_lr": args.matrix_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers.append(optimizer_lora)
```

### 5. Add hyperparameters (around line 115)

```python
peft_method = os.environ.get("PEFT_METHOD", "lora")
peft_rank = int(os.environ.get("PEFT_RANK", "0"))  # 0 = disabled
```

Pass `peft_rank` to `GPT.__init__`.

### 6. Count LoRA params in the log (around line 1427)

```python
n_lora = sum(p.numel() for m in [base_model.q_lora, base_model.v_lora]
             if m is not None for p in m.parameters())
log0(f"model_params:{n_params} lora_params:{n_lora} total:{n_params + n_lora}")
```

## Things to NOT forget

1. **Warmdown**: set `WARMDOWN_ITERS=1200` (already in queue.yaml). Phase 1 runs lacked this.

2. **LoRA in DDP**: the LoRA params are part of the model → DDP will sync them automatically.
   No special handling needed.

3. **Saving/loading**: the LoRA params are regular `nn.Parameter` in the model → `state_dict()`
   will include them. The compressed size will increase by the LoRA param count.

4. **Size budget check**: for 10L-5U-512d with rank 8 LoRA on Q and V:
   - Q LoRA: 10 steps × (8×512 + 512×8) = 10 × 8192 = 81,920 params
   - V LoRA: 10 steps × (8×256 + 256×8) = 10 × 4096 = 40,960 params (KV dim = 256 with 4 KV heads)
   - Total: ~123K params × 1 byte (int8) ≈ **0.12 MB** — negligible!
   - Even rank 16 is only ~0.24 MB. You have plenty of room.

5. **TTT interaction**: the existing TTT LoRA applies at eval time ON TOP of your training LoRA.
   They're complementary. Make sure `GPT.forward()` handles both:
   - During training: use per-step LoRA (your new adapters)
   - During TTT eval: use both per-step LoRA (frozen) + TTT LoRA (adapted per doc)

6. **Don't apply LoRA to K**: standard practice is Q+V only. K projection already benefits
   from RoPE and doesn't gain much from low-rank adaptation.

7. **Consider MLP too**: if Q+V LoRA shows promise, try adding a per-step MLP adapter.
   MLP is ~60% of layer FLOPs, so adaptation there could matter. But start with Q+V only.

## Param cost estimates

| Rank | Q+V LoRA (10 steps, d=512, kv=256) | As int8 MB |
|------|-------------------------------------|------------|
| 4    | ~61K params                          | ~0.06 MB   |
| 8    | ~123K params                         | ~0.12 MB   |
| 16   | ~245K params                         | ~0.24 MB   |
| 32   | ~491K params                         | ~0.47 MB   |

Base model is 8.51 MB. Even rank 32 keeps you well under 16 MB.

## Quick test

After implementing, verify with a short run:

```bash
cd /path/to/parameter-golf
PEFT_METHOD=lora PEFT_RANK=8 NUM_LAYERS=10 NUM_UNIQUE_LAYERS=5 MODEL_DIM=512 \
MAX_WALLCLOCK_SECONDS=120 WARMDOWN_ITERS=0 \
torchrun --standalone --nproc_per_node=4 experiments/phase2_PEFT/train_gpt_peft.py
```

Check that:
- `lora_params:` appears in the log with the expected count
- Loss goes down (no NaN or divergence)
- The LoRA doesn't slow down ms/step by more than 2-3%
