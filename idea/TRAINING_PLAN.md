# Training Plan — UT / MoE / Latent MoE / LoRA Experts

> Goal: systematic exploration of parameter-efficient architectures under the 16MB / 10min constraint.
> Even if nothing beats the baseline — you'll have built intuition for parameter budgets, training dynamics,
> and architectural trade-offs that most people only read about.

---

## Core Ideas

### 1. Universal Transformer (UT) — depth for free

**What it is:**
Instead of 10 distinct layers with 10 separate weight tensors, you have K unique layers reused N times each.
The total effective depth is K × N but you only store K layers worth of parameters.

```
Standard:   L1 → L2 → L3 → L4 → L5 → L6 → L7 → L8 → L9 → L10   (10 weight sets)
UT (A/B×5): A  →  B  →  A  →  B  →  A  →  B  →  A  →  B  →  A  →  B   (2 weight sets)
```

**The tied embedding analogy:**
Just like tied embeddings share the input and output matrix to halve embedding params,
UT shares layer weights across depth to multiply depth without multiplying params.

**Compute cost:** identical to a standard model with the same effective depth — you still run K×N forward passes.
The saving is purely in stored parameters → freed budget goes to wider hidden size or more steps.

**Critical detail — depth embedding:**
The shared layer has no way to know if it's on step 1 or step 7. You must add a learnable
depth embedding (one vector per step, shape: `[num_steps, hidden]`) added to x at each step.
Without this, all steps are indistinguishable and quality collapses ~0.03-0.05 bpb.

**Main risk:** harder to optimize. Two unique layers (A/B alternating) is a good middle ground —
A can specialize in attention, B in FFN. Single unique layer (1×10) is the hardest.

---

### 2. Standard MoE — capacity without width

**What it is:**
Replace the single FFN in each transformer layer with N expert FFNs.
A router network looks at each token and selects the best k experts for it.
Only k/N of the FFNs actually run per token — sparse computation.

```
Token → Router → [Expert 1, Expert 2, Expert 3, Expert 4]
                          ↑ top-1 selected, rest skipped
```

**Sparse (top-k) vs Dense (weighted average):**

| Mode | Compute | Gradient | Collapse risk |
|------|---------|----------|---------------|
| Top-1 sparse | cheapest | none through routing | high |
| Top-2 sparse | moderate | none through routing | moderate |
| Dense (weighted avg) | all experts run | clean, fully differentiable | none |

For **full-size experts**: use sparse (Top-1 or Top-2). Dense routing with full experts = N× compute = fewer training steps = net loss.

**Routing collapse:** the main failure mode. All tokens gravitate to 1 expert; the others get no gradient and die.
**Always add load balancing loss** from step 1: `total_loss += 0.01 * load_balance_loss`.

**LoRA init trick:** initialize LoRA_B = zeros so all experts start identical and diverge gradually.
If experts start random, the router gets random gradients early and learns nothing useful.

---

### 3. Latent MoE — many tiny experts

**What it is:**
Instead of N experts each operating in the full hidden space (d=512), project down to a smaller latent space first,
run experts there, then project back up. The down/up projections are shared across all experts.

```
x ∈ ℝ^512
    ↓  W_down  (shared, 512 → 128)
z ∈ ℝ^128
    ↓  router → N tiny experts, each a small FFN in ℝ^128
z' ∈ ℝ^128
    ↓  W_up  (shared, 128 → 512)
output ∈ ℝ^512
```

**Why it matters for parameter budget:**

| Setup | Experts | Params |
|-------|---------|--------|
| Standard MoE (d=512, mlp=1024) | 4 | ~4.0M |
| Latent MoE (d'=128, mlp=256) | 4 | ~0.3M |
| Latent MoE (d'=128, mlp=256) | **16** | ~0.7M |

Same param budget → 4× more experts → more routing diversity.

**Key insight:** with cheap experts you can afford **dense routing** (weighted average of all experts).
No routing collapse. No load balancing loss needed. Fully differentiable. Nearly free extra compute.

**Main risk:** information bottleneck. Experts only see what W_down lets through.
If d' is too small, experts are blind to important token features.
d'=128 is a reasonable starting point for d=512. Test d'=64 vs d'=256 to measure information loss.

---

### 4. LoRA Experts — delta-based specialization

**What it is:**
Keep one full shared FFN (base expert). Add N small LoRA adapters.
Each "expert" is: `base_FFN(x) + B_i(A_i(x))` where A_i ∈ ℝ^{r×d}, B_i ∈ ℝ^{d×r}.

```
x → shared_FFN(x)   +   router_weight_i × B_i(A_i(x))   (for all i, summed)
        full cost           tiny cost (rank r per adapter)
```

**Why dense routing is free here:**
A_i and B_i are tiny matrices. For rank=8, d=512: each adapter is 2 × 512 × 8 = 8K params and 2 matmuls of size (512×8).
Running 4 adapters densely adds ~4 × 2 × (512×8) ops vs the full FFN at ~2 × 512 × 1024 ops.
The adapter overhead is roughly 3% of the FFN cost. Dense routing is essentially free.

**Init:** A_i ~ small random, B_i = zeros. All experts start identical, diverge as training progresses.

**LoRA per UT step (variant):**
Instead of MoE routing between adapters, assign one LoRA adapter per UT step.
Step i always uses: `base_layer(x) + lora_i(x)`.
No routing needed. Cheap differentiation across UT depth.

---

## Parameter Budget Reference

Baseline: hidden=512, 10 layers, ~17M params, ~14.7MB artifact (int8 + fp16 embed)

Rough per-layer cost at different widths (attention + FFN, int8):

| hidden | mlp | ~MB/layer | Layers in ~13MB |
|--------|-----|-----------|-----------------|
| 512 | 1024 | ~1.3MB | 10 |
| 640 | 1280 | ~2.0MB | 6-7 |
| 768 | 1536 | ~2.9MB | 4-5 |
| 1024 | 2048 | ~5.2MB | 2-3 |

**Rule of thumb before every experiment:** estimate MB cost. If it doesn't fit in ~13MB (leaving ~2-3MB for embedding), don't run it.

---

## Controlled Experiment Settings

Always use these env vars for fair comparison across different hardware speeds:

```bash
ITERATIONS=600 WARMDOWN_ITERS=100 MAX_WALLCLOCK_SECONDS=0
```

This gives fixed step count regardless of step time. 600 steps with warmdown at 500.
Use 3 seeds (1337, 42, 7) for anything that looks promising. Single seed for early exploration.

---

## Phase 1 — Universal Transformer

Goal: find if weight tying helps and what unique-layer / width trade-off works best.

| ID | Unique layers | Steps | Eff. depth | Hidden | Est. params | Notes |
|----|--------------|-------|-----------|--------|-------------|-------|
| UT-0 | 10 | 1 | 10 | 512 | baseline | sanity check |
| UT-1 | 5 | 2 | 10 | 512 | same | gentle tying |
| UT-2 | 2 (A/B) | 5 | 10 | 512 | ~8M | A/B specialization |
| UT-3 | 1 | 10 | 10 | 512 | ~6M | extreme sharing |
| UT-4 | 2 (A/B) | 5 | 10 | 640 | ~12M | wider with saved params |
| UT-5 | 2 (A/B) | 5 | 10 | 768 | ~16M | push width limit |
| UT-6 | 2 (A/B) | 7 | 14 | 512 | ~10M | more depth instead |
| UT-7 | 3 | 4 | 12 | 640 | ~15M | middle ground |

**Recommended order:** UT-0 → UT-2 → UT-4 → UT-5. If UT-2 already loses badly vs UT-0, skip UT-4/5.

**What to look for:**
- Does UT-2 (A/B×5) match UT-0? → weight tying is viable
- Does UT-4 > UT-2? → width beats depth for this architecture
- Does UT-5 fit in 16MB? → check artifact size before running

**Don't forget:** add depth embedding to all UT configs.

---

## Phase 2 — UT + Per-Step LoRA

**Only run if Phase 1 produces a UT config within ~0.02 bpb of baseline.**

Take the best Phase 1 config. Add one LoRA adapter per step (not routing — fixed assignment).

| ID | Base | LoRA rank | Notes |
|----|------|-----------|-------|
| UT-LoRA-A | best UT | 4 | minimal params |
| UT-LoRA-B | best UT | 8 | sweet spot |
| UT-LoRA-C | best UT | 16 | more capacity |

Each step i uses: `base_layer_output + lora_A_i @ x` (standard LoRA forward).
No router. No routing collapse. Just cheap per-step differentiation.

**What to look for:** does per-step LoRA close the gap vs standard transformer?
If UT-LoRA-B > UT baseline, the idea is working.

---

## Phase 3 — MoE Alone (no UT)

Standard 10-layer transformer, MoE replacing FFN only (not attention).

### 3A — Standard sparse MoE

| ID | Experts | Top-k | Load balance α | Notes |
|----|---------|-------|----------------|-------|
| MoE-S1 | 2 | 1 | 0.01 | minimal |
| MoE-S2 | 4 | 1 | 0.01 | standard |
| MoE-S3 | 4 | 2 | 0.01 | denser |
| MoE-S4 | 8 | 2 | 0.01 | many experts |

### 3B — Latent MoE (dense routing)

| ID | Experts | d' | Notes |
|----|---------|-----|-------|
| MoE-L1 | 4 | 256 | low compression |
| MoE-L2 | 4 | 128 | medium compression |
| MoE-L3 | 8 | 128 | more experts |
| MoE-L4 | 16 | 64 | many tiny experts |
| MoE-L5 | 16 | 128 | wider latent |

**Key comparison:** MoE-L3 (8 experts, d'=128) vs MoE-S2 (4 full experts) at similar total params.
That's the direct test of whether latent design is worth it.

**What to look for:**
- Does compression hurt? → compare MoE-L1 vs MoE-L2 (same experts, different d')
- More experts or wider experts? → compare MoE-L3 vs MoE-L5

### 3C — LoRA experts (dense routing)

| ID | Experts | LoRA rank | Notes |
|----|---------|-----------|-------|
| MoE-R1 | 4 | 8 | cheap diversity |
| MoE-R2 | 8 | 8 | more adapters |
| MoE-R3 | 4 | 32 | higher rank |
| MoE-R4 | 8 | 16 | balanced |

All use dense routing (weighted average). No load balancing loss needed.

---

## Phase 4 — Pairwise Combinations

**Only run combinations where both components showed improvement individually.**

| ID | Combination | Notes |
|----|-------------|-------|
| COMBO-1 | UT-best + LoRA experts (dense) | depth sharing + cheap capacity |
| COMBO-2 | UT-best + Latent MoE | depth sharing + bottleneck experts |
| COMBO-3 | UT-best + per-step LoRA + LoRA experts | LoRA for steps AND experts |
| COMBO-4 | Standard transformer + Latent MoE + LoRA experts | latent + delta |

For COMBO-3: be careful with param budget. Two sets of LoRA (one per step, one per expert) can add up.
Calculate before running.

---

## Phase 5 — Full Architecture (if you're still having fun)

```
Input x
│
├── [UT] 2 unique layers (A/B), repeated 5 times
│    └── depth embedding added at each step
│
├── [Latent MoE] inside each layer's FFN slot:
│    W_down (512→128) → 8 experts in ℝ^128 → W_up (128→512)
│    Dense routing (weighted average)
│
└── [Per-step LoRA] on W_down/W_up (rank 8, one pair per step)
     Differentiates how the bottleneck projects at each UT step
```

Estimated param cost (very rough):
- 2 unique A/B layers (attn only): ~2 × 0.8M = 1.6M
- Shared W_down + W_up: 2 × 512 × 128 = 0.13M
- 8 latent experts (d'=128, mlp=256): 8 × 2 × 128 × 256 = 0.52M
- Per-step LoRA on W_down (rank 8, 10 steps): 10 × 2 × 512 × 8 = 0.08M
- Embedding (fp16, tied): ~25M × 0.5 byte ≈ depends on vocab
- **Total layers:** ~2.3M → easily fits, leaves a lot of room

This is underparameterized. Use saved params for wider hidden (768) or more latent experts.

---

## Decision Tree

```
Phase 1: UT
├── UT within 0.02 bpb of baseline?
│   ├── YES → Phase 2 (per-step LoRA)
│   └── NO  → skip UT in combinations, note result

Phase 3A: Sparse MoE
├── Any MoE config beats baseline?
│   ├── YES → note best config
│   └── NO  → try latent / LoRA variants anyway (3B, 3C)

Phase 3B: Latent MoE
├── Latent d' matters a lot?
│   ├── YES (d'=256 >> d'=128) → info loss is real, keep d' high
│   └── NO  → compression is free, use small d' for more experts

Phase 3C: LoRA experts
├── Beats sparse MoE at same param cost?
│   ├── YES → preferred over sparse for combinations
│   └── NO  → sparse routing is better, use it in combinations

Phase 4: Combinations
└── Combine only the winners from Phases 1-3
```

---

## What You'll Learn Regardless of Results

| Experiment | What you learn |
|------------|----------------|
| UT-0 vs UT-2 | how much tying hurts raw expressivity |
| UT-2 vs UT-4 | whether width > depth for this task |
| UT-2 vs UT-6 | whether more steps > wider at fixed params |
| MoE-L1 vs MoE-L2 | how much the bottleneck costs |
| MoE-L3 vs MoE-S2 | latent experts vs full experts (same params) |
| MoE-R2 vs MoE-S2 | LoRA experts vs full experts |
| COMBO-1 vs UT-best | whether MoE adds anything on top of UT |

Even if everything loses to the baseline by 0.1 bpb, you will have calibrated intuition for:
- How much information the bottleneck destroys at each d'
- How quickly sparse routing collapses without load balancing
- Whether width or depth wins for small language models
- The real cost of adding routing overhead to a tight training budget

That intuition is not written in papers. You build it by running the experiments.
