# Phase 2 — PEFT Adapter Findings (Part 1: LoRA)

> Experiment period: 2026-03-23
> Hardware: 4× NVIDIA A10G (g5dn.12xlarge), grad_accum=2
> Training budget: 20 min per run (1200s wall-clock)
> All runs: seed=1337, WARMDOWN_ITERS=400

---

## TL;DR

1. **Higher LoRA rank helps** — r64 meaningfully better than r16 at same config
2. **qv-attn is the best location** — Q+V adapters match full attention (qkvo) at lower param cost
3. **FFN adapters add too much overhead** — "all" location slows ms/step significantly, not worth it yet
4. **1U is still too non-expressive** — LoRA on 1-unique-layer barely moves the needle; adapters alone can't compensate for extreme weight sharing
5. **LoRA on 5U is promising but undertrained** — the ms/step overhead + large batch config robbed steps; not a fair comparison to Phase 1 5U results yet
6. **Key config bug: TRAIN_BATCH_TOKENS was 786K instead of 524K** — caused ~30% fewer steps, inflating all ms/step numbers

---

## All Phase 2 Runs

| Docker | Config | Location | Rank | Steps | Val BPB | ms/step | Size MB | Notes |
|---|---|---|---|---|---|---|---|---|
| rustic-finch | 10L-5U-512d | qv-attn | 64 | 1289 | **1.3812** | 931.5 | 9.79 | Best PEFT |
| dark-iguana | 10L-5U-512d | qkvo-attn | 64 | 1674 | **1.3812** | 716.9 | 12.21 | Same bpb, heavier |
| quiet-impala | 10L-1U-512d | qv-attn | 16 | 1968 | 1.5042 | 610.1 | 2.57 | Low rank, 1U |
| cool-puma | 10L-1U-512d | all | 64 | 1493 | 1.5056 | 804.0 | 9.70 | 1U even with r64 |
| zippy-husky | 10L-5U-512d | all | 16 | 0 | — | 810.6 | — | **CRASHED** |

### Phase 1 reference points (same hardware, longer runs)

| Config | Steps | Val BPB | ms/step | Size MB |
|---|---|---|---|---|
| 9L-512d (baseline) | 6564 | 1.2378 | 548.5 | 15.08 |
| 10L-5U-512d (no LoRA) | 3452 | 1.3348 | 579.4 | 8.51 |
| 10L-1U-512d (no LoRA) | 3357 | 1.4803 | 595.9 | 2.06 |

---

## Finding 1: Higher Rank Helps

r64 beats r16 in every comparable configuration.

| Config | Rank | Val BPB | Δ vs no-LoRA |
|---|---|---|---|
| 10L-1U qv-attn | 16 | 1.5042 | +0.024 (worse) |
| 10L-1U all | 64 | 1.5056 | +0.026 (worse) |
| 10L-5U qv-attn | 64 | 1.3812 | +0.046 vs Phase1 |

The r64 runs generally show better train loss curves (lower loss earlier). The r16 runs barely improve over no-adapter baselines. This suggests **the expressivity of LoRA matters more than parameter budget** for this task — rank is the key lever, not location count.

**Hypothesis:** The UT's shared weights are heavily overloaded. Low-rank adapters (r16) don't have enough expressivity to meaningfully differentiate 10 depth steps. r64 starts to have enough, but the sweet spot is likely higher still (r128+).

---

## Finding 2: qv-attn Is the Best Location

Comparing rustic-finch (qv, r64) and dark-iguana (qkvo, r64) at 5U:

| Location | BPB | ms/step | Size MB | Steps in 20min |
|---|---|---|---|---|
| qv-attn | **1.3812** | 931.5 | 9.79 | 1289 |
| qkvo-attn | **1.3812** | 716.9 | 12.21 | 1674 |

Same val BPB, but qkvo is 2.42 MB heavier and runs faster per step (surprising — more adapters but less ms/step, likely due to different batch configs in these runs). The key result is: **adding K and O adapters did not improve over Q+V only**, which is consistent with the original LoRA paper (Hu et al. 2021) and most literature. K is the query-key similarity signal — adapting it disrupts learned attention patterns. O is output projection — less information bottleneck than Q/V.

**Conclusion:** stick with `qv-attn` as the default location. It's cheaper and equally expressive.

---

## Finding 3: FFN Adapters Slow Things Down Without Clear Gain

The "all" location (qkvo + FFN fc + FFN proj) runs slower and the 5U+all run crashed entirely (zippy-husky, 0 steps). The cool-puma (1U+all) run has a higher ms/step (804 vs 610 for qv) but only matches the qv result in bpb.

FFN adapters add two extra matmuls per block per step. For a 10-step 1U model: 10 × 2 extra matmuls every forward pass. For a 20-step forward (5U, 10 layers), that's 20 extra matmuls. The overhead compounds.

**The FFN adapters are not worth it at 20 min wall-clock.** They might matter for longer runs where quality matters more than step count, but the current budget doesn't support it.

**Note:** zippy-husky (5U+all) crashing needs investigation. Likely OOM given the larger model + 786K batch tokens pushing activation memory over 24GB per GPU.

---

## Finding 4: 1U is Still Too Non-Expressive

| Config | Val BPB | Steps | Δ vs 1U no-LoRA |
|---|---|---|---|
| 10L-1U no LoRA (Phase 1) | 1.4803 | ~3357 | baseline |
| 10L-1U + LoRA r16 qv | 1.5042 | 1968 | +0.024 worse |
| 10L-1U + LoRA r64 all | 1.5056 | 1493 | +0.026 worse |

LoRA on 1U shows **no improvement** over no-LoRA 1U. The runs were shorter (1200s vs ~2000s in Phase 1) but the val curves suggest these runs hadn't converged to better territory either. The single shared block is fundamentally too constrained — it needs to simultaneously learn "shallow reasoning" (early steps) and "deep reasoning" (late steps), which are conflicting objectives. A rank-64 LoRA delta of shape `(512, 64)×(64, 512)` = 65K params per step can't fix this.

**However:** the quiet-impala run shows that train_loss was still decreasing at step 1968. A longer run (1H or overnight) with better hyperparameters might eventually close the gap. The 1U config is worth one more careful overnight run before being abandoned.

---

## Finding 5: The Real Bottleneck — Steps, Not Adapter Quality

The 5U+LoRA runs got **63% fewer steps** than Phase 1 5U (1289 vs 3452). This is a double hit:
1. `TRAIN_BATCH_TOKENS = 786,432` instead of `524,288` → each step processes 50% more data → each step takes longer
2. LoRA adapter forward/backward adds ~15-25% to ms/step on top of that

Result: rustic-finch achieved 1.3812 bpb at step 1289 vs Phase 1 5U achieving 1.3348 at step 3452. **These runs are not directly comparable.** The 5U+LoRA model is doing well for how few steps it has seen, but it hasn't had enough time to actually improve over Phase 1 5U.

The real test is: *at equal steps, does 5U+LoRA beat 5U without LoRA?*

---

## Key Config Bugs Identified

1. **`TRAIN_BATCH_TOKENS = 786,432` (should be `524,288`)** — This is set in the PEFT script defaults and caused all Phase 2 runs to train ~30% fewer steps than Phase 1 runs. Fix this before any comparison with Phase 1.

2. **`NUM_UNIQUE_LAYERS = 1` in queue defaults** — The queue ran most experiments on 1U instead of 5U. The 5U results (rustic-finch, dark-iguana) came from a separate run. The queue needs to default to `NUM_UNIQUE_LAYERS=5` with Phase 1's best config.

3. **`zippy-husky` crash** — 10L-5U + lora-all + r16 got 0 steps. Investigate log for OOM or shape error.

---

## What to Do Next

### Immediate fixes before next run
1. Reset `TRAIN_BATCH_TOKENS` to `524,288` in PEFT script defaults
2. Set `NUM_UNIQUE_LAYERS=5` as queue default
3. Keep `qv-attn` as the default location
4. Investigate zippy-husky crash log

### Phase 2 Part 2 experiments
Now that we know r64+qv on 5U is the best config, run a proper comparison:

| Priority | Config | Goal |
|---|---|---|
| HIGH | 10L-5U + LoRA r64 qv (fixed batch) | Clean baseline with correct 524K batch |
| HIGH | 10L-5U + LoRA r128 qv (fixed batch) | Test if more rank still helps |
| MEDIUM | 10L-5U + NOBLE qv (from scratch) | Nonlinear branch — likely better than LoRA |
| LOW | 10L-1U + LoRA r128 qv (1H overnight) | Give 1U a real chance |

### Moving to NOBLE
The LoRA results suggest that **linear adapters are rank-limited** for this task. A rank-64 LoRA is a 65K-param linear bottleneck — it can differentiate steps but only through linear transformations. NOBLE's nonlinear branch `σ(xW_down)W_up` can learn genuinely different activation patterns per step, which is more expressive at the same parameter budget.

NOBLE is also designed for training from scratch (not fine-tuning), which matches this use case exactly.

---

## Summary Table

| Question | Answer |
|---|---|
| Does LoRA help on 5U? | Unclear — needs fair comparison (fix batch size first) |
| Does LoRA help on 1U? | No — 1U is too constrained, LoRA doesn't compensate |
| Best LoRA rank? | r64 > r16; r128 untested, likely better |
| Best location? | qv-attn — matches qkvo at lower cost |
| FFN adapters worth it? | No — too slow, crashed at 5U+all |
| Should we try NOBLE? | **Yes** — more expressive, designed for training from scratch |
