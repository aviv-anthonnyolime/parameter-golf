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
| ------|---------|----------|---------------|
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

```markdown
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

## Phase 1 — Universal Transformer [COMPLETED]

> Full results in `idea/PHASE1_UT_FINDINGS.md`

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

### Phase 1 Results

**Key outcome: weight tying works, but no UT config beat the baseline on wall-clock.**

| Config | Val Loss | Val BPB | ms/step | Size MB | Verdict |
|--------|----------|---------|---------|---------|---------|
| **Baseline 9L-512d** | **2.0900** | **1.2378** | **548.5** | **15.08** | **Winner** |
| 10L-5U-768d | 2.1992 | 1.3025 | 1054.9 | 18.57 | Best per-step BUT over 16MB, 2x slower |
| 14L-7U-640d | 2.2099 | 1.3088 | 1209.5 | — | Crashed (TTT OOM) |
| 12L-6U-512d | 2.2387 | 1.3259 | 689.0 | 10.19 | Best 512d UT |
| 10L-5U-512d | 2.2538 | 1.3348 | 579.5 | 8.51 | Best speed/quality trade-off |
| 10L-1U-512d | 2.4994 | 1.4803 | 595.9 | 2.06 | Extreme sharing hurts |

**Why baseline wins:** it's the fastest (548 ms/step → most steps in the time budget). UT saves params but the freed budget goes to wider dims → slower ms/step → fewer training steps.

**Findings:**

- 5+ unique layers is the minimum. 1-2 unique layers lose badly.
- Width (768d) wins per-step but ms/step penalty negates the gain.
- Rankings are stable from step 1000 — 20 min is enough to screen.
- 10L-5U-768d is over 16MB (18.57 MB) — needs aggressive quant or MLP reduction to fit.
- H100 projection (α=0.06): baseline still wins due to 13,780 steps in 10 min.

### The 768d Question

10L-5U-768d has the best loss-per-step of any UT config, but two problems:

1. **Over 16MB** (18.57 MB compressed)
2. **2x slower** than baseline (1055 vs 549 ms/step)

**Possible rescue paths for 768d:**

- Reduce MLP 2x → 1.5x (saves ~25% MLP FLOPs + ~25% params → may fit in 16MB)
- Reduce KV heads 4 → 2 (saves attention compute)
- Reduce effective layers 10 → 8 (saves 20% compute)
- Int6 quantization for MLP weights + int8 for attention (saves ~2-3 MB)
- Combined: 768d, 1.5x MLP, 2 KV heads, 8L-4U → estimated ~700 ms/step, ~14 MB

**Decision:** park 768d for now. Carry forward **10L-5U-512d** (8.51 MB, 579 ms/step) as the Phase 2 base. If PEFT adapters are small enough, revisit 768d with MLP reduction as a Phase 2 variant.

---

## Phase 2 — UT + Per-Step PEFT Adapters

Goal: differentiate shared UT layers cheaply using parameter-efficient fine-tuning methods.
Each UT step i gets its own adapter: `output = base_layer(x) + adapter_i(x)`.

### Base configs

Two bases to test on:

- **Primary: 10L-5U-512d** — fast (579 ms/step), fits easily (8.51 MB), room for adapters
- **Stretch: 10L-5U-768d (trimmed)** — 1.5x MLP, 2 KV heads to reduce ms/step and size

Both must run **with warmdown** (Phase 1 runs lacked warmdown, giving baseline an unfair edge).

### PEFT Methods to Compare

Test each method as a per-step adapter on the primary base. One adapter per UT step (10 adapters total for 10L). All applied to both attention projections and MLP.

| Method | Key Idea | Params/adapter (rank 8, d=512) | ms/step overhead | Why test it |
|--------|----------|-------------------------------|-----------------|-------------|
| **LoRA** | Low-rank A×B decomposition | ~8K per matrix | ~1-2% | Baseline PEFT, well understood |
| **AdaLoRA** | Adaptive rank allocation via SVD | ~8K (adapts during training) | ~3-5% | May allocate rank where it matters most |
| **DoRA** | Weight-decomposed LoRA (magnitude + direction) | ~8K + magnitude vector | ~2-3% | Better training dynamics than LoRA |
| **MoRA** | High-rank via square matrices with compression | ~8K but effective rank higher | ~3-5% | More expressivity at same param count |
| **VeRA** | Shared random projections + per-layer scaling | ~1K (just scaling vectors) | ~1% | Extreme param efficiency, almost free |
| **LoHa** | Low-rank Hadamard product | ~8K per matrix | ~2-3% | Captures multiplicative interactions |

> You will fill in the paper links, GitHub repos, and implementation details in the table below after reviewing each method.

### PEFT Reference Table (to be completed)

| Order | Method | Main idea | Canonical paper | Relevance for your Universal Transformer setup | Example implementation(s) |
|---|---|---|---|---|---|
| 1 | **LoRA** | Adds a low-rank residual update to a weight matrix instead of a full dense extra matrix. | **Hu et al., 2021 — _LoRA: Low-Rank Adaptation of Large Language Models_**  [oai_citation:0‡arXiv](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com) | **Very relevant baseline.** Best first test for adding per-layer or per-step extra capacity to a shared UT block because it is simple, stable, and easy to port to your own layers.  [oai_citation:1‡arXiv](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com) | **microsoft/LoRA** (`loralib`)  [oai_citation:2‡GitHub](https://github.com/microsoft/LoRA?utm_source=chatgpt.com) |
| 2 | **NOBLE** | Adds a **nonlinear low-rank branch** to linear layers: \( \sigma(xW_{down})W_{up} \). Designed as a permanent architectural component for training from scratch. | **Smith, 2026 — _NOBLE: Accelerating Transformers with Nonlinear Low-Rank Branches_** | **Extremely relevant.** Very close to your goal because you want extra per-layer / per-step expressiveness during training, not PEFT on a frozen model. | **ethansmith2000/noble** |
| 3 | **DoRA** | Splits the weight behavior into **magnitude** and **direction**, and applies LoRA-style adaptation mainly to the direction. | **Liu et al., 2024 — _DoRA: Weight-Decomposed Low-Rank Adaptation_**  [oai_citation:3‡arXiv](https://arxiv.org/pdf/2402.09353?utm_source=chatgpt.com) | **Very relevant.** Good next test if plain LoRA feels too restrictive and you want richer geometry in the shared recurrent block.  [oai_citation:4‡arXiv](https://arxiv.org/pdf/2402.09353?utm_source=chatgpt.com) | **NVlabs/DoRA**  [oai_citation:5‡GitHub](https://github.com/NVlabs/DoRA?utm_source=chatgpt.com), **nbasyl/DoRA**  [oai_citation:6‡GitHub](https://github.com/nbasyl/DoRA?utm_source=chatgpt.com) |
| 4 | **MoRA** | Replaces LoRA's low-rank bottleneck with a **higher-rank update** while keeping a similar trainable parameter budget. | **Jiang et al., 2024 — _MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning_**  [oai_citation:7‡arXiv](https://arxiv.org/abs/2405.12130?utm_source=chatgpt.com) | **Probably the most aligned with your goal.** Your target is more expressiveness inside a shared block, and MoRA was proposed exactly because low-rank updates can be too limiting.  [oai_citation:8‡arXiv](https://arxiv.org/abs/2405.12130?utm_source=chatgpt.com) | **kongds/MoRA**  [oai_citation:9‡GitHub](https://github.com/kongds/MoRA?utm_source=chatgpt.com) |
| 5 | **LoHa** | Uses low-rank factors combined with a **Hadamard product** to obtain a more expressive update than plain LoRA. | In practice it is commonly tied to the **LyCORIS** ecosystem; HF PEFT explicitly links LoHa to the **FedPara** Hadamard-style parameterization, and the LyCORIS paper documents LoHa/LoKr as implemented methods.  [oai_citation:10‡Hugging Face](https://huggingface.co/docs/peft/main/package_reference/loha?utm_source=chatgpt.com) | **Interesting experimental option.** Worth testing if you want “more expressive than LoRA” without jumping to a full dense extra matrix.  [oai_citation:11‡Hugging Face](https://huggingface.co/docs/peft/main/package_reference/loha?utm_source=chatgpt.com) | **HF PEFT LoHa**  [oai_citation:12‡Hugging Face](https://huggingface.co/docs/peft/main/package_reference/loha?utm_source=chatgpt.com), **LyCORIS** docs/examples  [oai_citation:13‡GitHub](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Demo.md?utm_source=chatgpt.com) |
| 6 | **AdaLoRA** | Learns how to **reallocate rank budget** across layers/modules instead of keeping the same rank everywhere. | **Zhang et al., 2023 — _AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning_**  [oai_citation:14‡arXiv](https://arxiv.org/abs/2303.10512?utm_source=chatgpt.com) | **Moderately relevant.** More useful if you want the model to learn which UT modules or recurrent steps deserve more capacity; less essential as a first expressiveness benchmark.  [oai_citation:15‡arXiv](https://arxiv.org/abs/2303.10512?utm_source=chatgpt.com) | **QingruZhang/AdaLoRA**  [oai_citation:16‡GitHub](https://github.com/QingruZhang/AdaLoRA?utm_source=chatgpt.com), also merged into **HF PEFT** according to the repo.  [oai_citation:17‡GitHub](https://github.com/QingruZhang/AdaLoRA?utm_source=chatgpt.com) |
| 7 | **VeRA** | Shares a pair of random low-rank matrices across layers and learns only small scaling vectors. | **Kopiczko et al., 2023/2024 — _VeRA: Vector-based Random Matrix Adaptation_**  [oai_citation:18‡arXiv](https://arxiv.org/abs/2310.11454?utm_source=chatgpt.com) | **Less relevant for your main objective.** Good when trainable/storage budget is the main pain point, but less compelling when your priority is maximizing expressiveness in a shared block.  [oai_citation:19‡arXiv](https://arxiv.org/abs/2310.11454?utm_source=chatgpt.com) | **HF PEFT VeRA**  [oai_citation:20‡Hugging Face](https://huggingface.co/docs/peft/package_reference/vera?utm_source=chatgpt.com) |

### Adapter Location Options

Each PEFT method can target different subsets of projections. Use the `adapter_location` parameter to control this:

| `adapter_location` | Q | K | V | O | MLP fc | MLP proj | Use case |
|---|---|---|---|---|---|---|---|
| `"qv-attn"` | yes | - | yes | - | - | - | Classic LoRA (Hu et al.) — minimal, cheapest |
| `"qkvo-attn"` | yes | yes | yes | yes | - | - | Full attention adaptation |
| `"ffn"` | - | - | - | - | yes | yes | MLP-only adaptation |
| `"qv-attn-ffn"` | yes | - | yes | - | yes | yes | Q/V + MLP — good default |
| `"all"` | yes | yes | yes | yes | yes | yes | Everything — most params, most expressive |

**Recommendation:** start with `"qv-attn"` (the original LoRA targets) for rank sweeps, then test `"qv-attn-ffn"` to see if MLP adapters add value. Only use `"all"` if param budget allows — K and O adapters have diminishing returns in most settings.

### Experiment Plan

**Step 1 — LoRA baseline (rank sweep):**

| ID | Base | Method | Rank | Notes |
|----|------|--------|------|-------|
| PEFT-L4 | 10L-5U-512d | LoRA | 4 | minimal |
| PEFT-L8 | 10L-5U-512d | LoRA | 8 | sweet spot expected |
| PEFT-L16 | 10L-5U-512d | LoRA | 16 | more capacity |

All with warmdown. This establishes the LoRA baseline to beat.

**Step 2 — Method comparison (fixed rank 8):**

| ID | Base | Method | Rank | Notes |
|----|------|--------|------|-------|
| PEFT-AL | 10L-5U-512d | AdaLoRA | 8 (initial) | adaptive rank — does it help? |
| PEFT-DO | 10L-5U-512d | DoRA | 8 | better training stability? |
| PEFT-MO | 10L-5U-512d | MoRA | 8 | higher effective rank? |
| PEFT-VE | 10L-5U-512d | VeRA | — | minimal params — how far can you go? |
| PEFT-LH | 10L-5U-512d | LoHa | 8 | multiplicative interactions |

Compare all at rank 8 (or equivalent param budget for VeRA). 20 min runs to screen, 1H for the winner.

**Step 3 — Winner on 768d trimmed (if any PEFT beats baseline):**

| ID | Base | Method | Rank | Notes |
|----|------|--------|------|-------|
| PEFT-768-W | 10L-5U-768d-trim | winner | best rank | 1.5x MLP, 2 KV heads, with warmdown |

This tests whether the 768d wider model + PEFT + size reduction can finally beat the baseline.

### What to look for

- **Does any PEFT close the gap to baseline?** UT alone is ~0.065 bpb behind. If PEFT recovers 0.03+, it's working.
- **VeRA vs LoRA:** VeRA uses 8x fewer params. If quality is close, it's the winner for this param-constrained setup.
- **DoRA vs LoRA:** DoRA should train more stably with weight decomposition. Check if it converges faster (matters for short runs).
- **MoRA:** if the compression trick works, it gets higher effective rank at the same param budget — could be significant for small models.
- **ms/step overhead:** anything >5% overhead is a concern. VeRA and LoRA should be <2%.

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
