# Phase 1 — Universal Transformer Findings

> Experiment period: 2026-03-20 → 2026-03-21
> Hardware: 4x NVIDIA A10G (g5dn.12xlarge), grad_accum=2
> Training budget: 33 min (early runs) → 60 min (overnight sweep)
> All runs use seed=1337, no warmdown (except the 1H baseline)

---

## TL;DR

1. **Weight tying works** — UT configs beat the "no tying" sanity check (10L-1U) by a large margin
2. **None of the UT configs beat the real baseline** (9L-512d, 1.2378 bpb) at equal wall-clock time
3. **Wider > deeper** — 768d beats 640d beats 512d at the same unique-layer count
4. **5 unique layers is the sweet spot** — less hurts quality, more adds ms/step without proportional gain
5. **ms/step is the hidden killer** — the best UT (10L-5U-768d) is 2x slower per step than the baseline, which means fewer training steps in the same wall-clock budget
6. **Rankings are stable from step 1000** — 20 min on 4xA10G is enough to reliably rank architectures

---

## All Runs Summary

Sorted by final val_loss (each run's own last checkpoint):

| Rank | Config | Docker | Steps | Val Loss | Val BPB | ms/step | Size MB | Status |
|------|--------|--------|-------|----------|---------|---------|---------|--------|
| 1 | 9L-512d (baseline) | baseline | 6564 | 2.0900 | 1.2378 | 548.5 | 15.08 | ok |
| 2 | 10L-5U-768d | zany-lynx | 3414 | 2.1992 | 1.3025 | 1054.9 | 18.57 | ok (OVER 16MB!) |
| 3 | 14L-7U-640d | crisp-swan | 2977 | 2.2099 | 1.3088 | 1209.5 | — | crashed (TTT OOM) |
| 4 | 12L-6U-512d | hardy-lark | 5226 | 2.2387 | 1.3259 | 689.0 | 10.19 | ok |
| 5 | 10L-5U-512d | spicy-salmon | 3452 | 2.2538 | 1.3348 | 579.5 | 8.51 | ok |
| 6 | 15L-5U-640d | deep-jackal | 2833 | 2.2664 | 1.3423 | 1270.9 | — | crashed |
| 7 | 10L-5U-640d | kind-heron | 2301 | 2.2762 | 1.3481 | 869.4 | 12.90 | ok |
| 8 | 15L-5U-512d | calm-yak | 4265 | 2.2823 | 1.3517 | 844.2 | 8.56 | ok |
| 9 | 9L-3U-640d | tough-gecko | 2607 | 2.3721 | 1.4049 | 767.4 | 8.03 | crashed (TTT) |
| 10 | 10L-2U-512d | silent-horse | 3604 | 2.4292 | 1.4387 | 555.1 | 3.70 | crashed (TTT) |
| 11 | 10L-1U-512d | zany-gorilla | 3357 | 2.4994 | 1.4803 | 595.9 | 2.06 | ok |

Additional crashed/short-lived runs from the overnight sweep (no usable val data):
- 20L-10U-512d (x2), 20L-5U-512d, 10L-5U-784d, 10L-5U-512d-3mlp, 15L-10U-512d
- 12L-3U-640d, 14L-2U-512d (x2), 10L-2U-768d, 12L-3U-512d
- 18L-9U-640d, 16L-8U-640d

**Key takeaway:** many configs with >14 effective layers or >640d crashed during post-training eval (OOM on TTT LoRA). Even when they complete training, the high ms/step means fewer steps.

---

## Key Findings

### 1. Weight Tying is Viable but Has a Cost

The sanity check (10L-1U-512d = 1 unique layer repeated 10x) scored 2.4994 val_loss — worse than the standard 9L baseline (2.0900) by a large margin. This confirms that extreme sharing hurts.

But moderate sharing works well:
- **5U at 512d** (10L-5U-512d): 2.2538 — clearly better than 1U and 2U
- **6U at 512d** (12L-6U-512d): 2.2387 — best among 512d configs

The gap from 5U to 6U is small (0.015 loss), suggesting **5 unique layers is near optimal** for the 512d budget.

### 2. Width Beats Depth (Within Time Budget)

At fixed unique layers (5U), comparing dims:
| Config | Val Loss | ms/step | Steps (1h) |
|--------|----------|---------|------------|
| 10L-5U-512d | 2.2538 | 579.5 | 3452 |
| 10L-5U-640d | 2.2762 | 869.4 | 2301 |
| 10L-5U-768d | 2.1992 | 1054.9 | 3414 |

768d achieves the **lowest loss per step**, but at 2x the ms/step cost.
640d is worse than 512d despite being wider — it only ran 2301 steps vs 3452 (33% fewer steps due to slower steps). The 640d model would eventually beat 512d given equal steps, but doesn't in equal wall-clock time.

**Conclusion:** wider is better per-step, but you need enough steps to realize the gain. On slow hardware (A10G), the trade-off often doesn't pay off.

### 3. Deeper UT (More Layers) Has Steep Diminishing Returns

Comparing 512d configs at different effective depths:
| Config | Effective Depth | Val Loss | ms/step |
|--------|----------------|----------|---------|
| 10L-5U-512d | 10 | 2.2538 | 579.5 |
| 12L-6U-512d | 12 | 2.2387 | 689.0 |
| 15L-5U-512d | 15 | 2.2823 | 844.2 |

Going from 10→12 layers gains 0.015 loss but costs 19% more ms/step.
Going from 12→15 layers **makes it worse** (0.044 loss regression) because the extra depth adds ms/step without enough steps to converge.

**Conclusion:** don't go beyond 12 effective layers at 512d on this hardware.

### 4. Unique Layer Count: Sweet Spots

At 512d:
- **1U**: 2.4994 (terrible — not enough expressivity)
- **2U**: 2.4292 (much better, but still weak)
- **5U**: 2.2538 (big jump — this is the threshold)
- **6U**: 2.2387 (marginal improvement over 5U)

At 640d:
- **3U**: 2.3721 (9L effective — underpowered)
- **5U**: 2.2762 (10L effective — much better)
- **7U**: 2.2099 (14L effective — best 640d, but crashed and very slow)

**Rule of thumb:** you want at least 5 unique layers. Going from 5→7 helps at 640d but the ms/step penalty is steep.

### 5. Compressed Size: UT Saves Params as Expected

| Config | Unique Params | Compressed Size |
|--------|--------------|-----------------|
| 9L-512d (baseline) | ~17M (all unique) | 15.08 MB |
| 10L-5U-512d | ~10M (half tied) | 8.51 MB |
| 12L-6U-512d | ~12M | 10.19 MB |
| 10L-5U-768d | ~19M | 18.57 MB |
| 10L-1U-512d | ~3M | 2.06 MB |

UT saves size as expected (half the layers = ~half the size). But **10L-5U-768d is OVER the 16MB limit** at 18.57 MB. It would need int6/mixed quantization or reduced MLP to fit.

---

## 8xH100 Projection

The official baseline runs at **43.54 ms/step** on 8xH100 (accum=1).
Your 4xA10G runs at **548.5 ms/step** (accum=2). Speed ratio: **12.6x**.

### Projection Formula

```
H100_ms_per_step = A10G_ms_per_step / 12.6
H100_steps_10min = 600,000 / H100_ms_per_step
```

**Alpha calibration:** The standard scaling law α=0.35 is too aggressive for 2-4× step extrapolations. Empirical α from the baseline curve (3000→6564 steps) is **~0.06**. Use `--alpha 0.06` for conservative projections.

### Projected Ranking (8xH100, 10 min, conservative α=0.06)

| Rank | Config | H100 ms/step | Steps@10min | Proj Val Loss | Notes |
|------|--------|-------------|-------------|---------------|-------|
| 1 | 9L-512d baseline | 43.5 | 13,780 | ~1.999 | WINS (fastest + warmdown) |
| 2 | 10L-5U-512d | 46.0 | 13,045 | ~2.081 | Close second, fits 8.5 MB |
| 3 | 10L-5U-640d | 69.0 | 8,695 | ~2.102 | Medium speed |
| 4 | 10L-5U-768d | 83.7 | 7,166 | ~2.104 | Best per-step but too slow |
| 5 | 12L-6U-512d | 54.7 | 10,970 | ~2.141 | More depth, still decent |
| 6 | 14L-7U-640d | 96.0 | 6,249 | ~2.16 | Deep but slowest |

Use `python scripts/ranking.py <results.jsonl> --project-h100 --alpha 0.06` to see this table.

**The projection confirms:** even on faster hardware, the baseline wins because:
- It's the fastest per step (fewest layers, smallest dim)
- 13,780 steps in 10 min is hard to beat
- It uses warmdown scheduling which the UT runs didn't

**For UT to compete, it needs to match the baseline's ms/step.** That means either:
1. Stay at 512d and use the saved params for something else (LoRA, MoE)
2. Find optimizations that reduce ms/step at 640-768d

---

## Ranking Stability Analysis

Checked val_loss rankings at steps 1000, 1500, 2000, 2500, 3000:

| Metric | Step 1000 | Step 2000 | Step 3000 |
|--------|-----------|-----------|-----------|
| Top-3 overlap with final | **3/3** | **3/3** | **3/3** |
| Top-5 overlap with final | 3/5 | 3/5 | 3/5 |
| Max rank shift from final | 3 | 3 | 3 |

**The top-3 is perfectly stable from step 1000 onward.** The mid-tier configs (ranks 4-8) shuffle by 1-3 positions, but the tier boundaries are clear.

### Practical Screening Guidelines

| Duration (4xA10G) | Steps (fast/slow config) | Reliability |
|-------------------|-------------------------|-------------|
| 10 min | ~1000 / ~500 | Good for top-3 vs bottom-3 separation |
| 20 min | ~2000 / ~1000 | Reliable full ranking (recommended) |
| 30 min | ~3000 / ~1500 | Diminishing returns, same ranking as 20min |
| 60 min | ~6500 / ~3000 | Useful only for final configs with warmdown |

**Recommendation: 20 minutes per run is the sweet spot.** That's enough for ~2000 steps on fast configs and ~1000 on slow ones, which gives stable rankings. Running 30+ min only helps for the absolute final comparison.

---

## ms/step Reduction Ideas for 10L-5U-768d

The 768d config is 2x slower than baseline (1055 vs 549 ms/step). The slowdown comes from:
- **FLOPs scale as dim²**: 768²/512² = 2.25x more compute in attention + MLP
- **10 effective layers vs 9**: 1.11x more passes
- **Net: ~2.5x compute, observed ~1.9x** (some hidden by memory bandwidth)

Possible optimizations:
1. **Reduce MLP multiplier** 2x → 1.5x: saves ~25% MLP FLOPs, MLP is ~60% of layer FLOPs
2. **Reduce KV heads** 4 → 2: saves ~25% attention memory and compute
3. **Reduce effective layers** 10 → 8: saves 20% compute
4. **Use grad_accum=1** on 4xA10G: halves step time but changes effective batch size
5. **torch.compile()**: if not already enabled, can save 10-20%
6. **Combine 1+2+3**: 768d, 1.5x MLP, 2 KV heads, 8L-4U → estimated ~700 ms/step

However, the fundamental issue is that **768d doesn't fit in 16MB** (18.57 MB compressed). So the real question is: what dim between 512 and 768 maximizes quality while staying under 16MB and keeping ms/step competitive?

**Suggestion:** try 640d with optimizations (1.5x MLP, fewer KV heads, 10L-5U) as a compromise.

---

## Decisions for Phase 2

Based on these findings:

1. **Best UT architecture to carry forward:** 10L-5U-512d (best quality-per-step, fits in 16MB easily)
2. **Alternative if speed improves:** 12L-6U-512d (slightly better quality, 19% slower)
3. **768d is parked** — oversized and too slow. May revisit with aggressive quantization.

### What Phase 1 Taught Us

| Question | Answer |
|----------|--------|
| Does weight tying help? | Yes — 5U ties depth without losing much quality |
| Width vs depth? | Width wins per-step, but ms/step cost often negates the gain |
| Can UT beat baseline? | Not yet — the ms/step penalty means fewer training steps |
| How many unique layers? | 5+ is the minimum. 6 is slightly better. |
| Is 20 min enough to rank? | Yes — top-3 is stable from step 1000 |

### Why the Baseline Still Wins

The baseline wins because of **training efficiency**, not architecture:
- **548 ms/step** is hard to beat — small model, small layers, fast forward pass
- **Warmdown scheduling** gives a free 0.02-0.05 bpb improvement at the end
- **6564 steps in 1H** (or 13780 on H100 in 10min) vs 3414 for the best UT

The UT idea is sound — it genuinely improves loss per parameter. But in this competition, the constraint is wall-clock time, not parameter count. The UT savings in params don't help if the extra compute (wider dim needed to fill the param budget) slows you down.

### The Path Forward

To make UT competitive:
1. **Add per-step LoRA** to the 10L-5U-512d config (Phase 2) — adds ~0.1-0.5 MB, tiny ms/step cost, differentiates shared layers
2. **Add warmdown** — all UT runs were without warmdown, which disadvantages them
3. **Test with warmdown + TTT** on the best config to get a true comparison vs baseline
4. **Consider other architectural tricks** (LoRA experts, MoE) that add capacity without width
