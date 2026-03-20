# Parameter Golf – Idea Development Log

## Initial Idea

I want to reduce parameter count without reducing capability by combining:
- Universal Transformer (parameter sharing across depth)
- Mixture of Experts (sparse routing)

Idea:
A "2-side Universal MoE Transformer" that improves performance under constraints.

---

## Refinement 1 – UT + MoE

Concept:
- Use a Universal Transformer (same layer reused iteratively)
- Add MoE routing inside each step

Goal:
- Replace depth with recurrence
- Maintain capacity via sparse experts

Limitation identified:
- MoE still requires multiple **static experts**
- Parameter cost still scales with number of experts

---

## Refinement 2 – Core Innovation (Dynamic Experts via LoRA)

New idea:

Instead of storing multiple experts:
- Use **one base expert**
- Use **LoRA adapters** to generate variations

Process:
1. Base weight is shared
2. Multiple LoRA modules are trained
3. For each input:
   - Combine LoRA adapters dynamically
   - Create "virtual experts"
4. Route tokens between these generated experts

Key insight:
> Experts are not stored — they are **generated on the fly**

---

## Refinement 3 – “2-Side” Definition

“2-side” means two levels of adaptation:

1. **Expert creation**
   - Dynamic combination of LoRA adapters

2. **Expert selection**
   - MoE-style routing of tokens

So:
> The model dynamically creates experts AND selects them

---

## Refinement 4 – Integration with MoEUT

Reference:
MoEUT (Mixture-of-Experts Universal Transformers)

MoEUT:
- Uses shared layers (UT)
- Adds static experts (MoE)

Limitation:
- Experts are still fixed and memory-heavy

Our extension:
- Replace static experts with dynamic LoRA-generated experts

---

## Final Proposal (Short Version, <750 chars)

**Dynamic LoRA Experts for Universal MoE Transformers**

Building on MoEUT (Mixture-of-Experts Universal Transformers), I propose replacing static experts with dynamically generated ones. Instead of storing multiple expert networks, the model uses a single shared layer (as in Universal Transformers) and a small set of LoRA adapters. For each input, the model combines these adapters to create specialized experts on the fly, then routes tokens between them. This “two-sided” approach—creating and selecting experts dynamically—keeps parameter count very low while enabling flexible, input-dependent capacity and iterative refinement across steps.

---

## Baseline Understanding (Parameter Golf Repo)

### Constraints:
- ~16MB model size
- ~10 minute training
- Very limited compute

### Baseline Model:
- Layers: 9
- Width: 512
- Heads: 8 (4 KV heads)
- MLP: 2× expansion
- Context: 1024
- Batch tokens: 524,288
- Iterations: 20,000

### Learning Rates:
- Embedding: 0.6
- Head: 0.008
- Matrix: 0.04
- Scalar: 0.04
- Tied embedding: 0.05

### Size:
- ~17M parameters
- ~15.8MB saved model

---

## Other Submissions Insights

### 10-layer models:
- Push depth slightly higher
- Compensate using quantization (int6 / int8 mix)

### Improvements used:
- Sliding window evaluation
- Mixed precision
- FP16 embeddings
- Weight decay tuning (Muon)
- Better initialization (Overtone)

Key takeaway:
> Gains come from squeezing efficiency, not adding parameters

---

## Final Concept Summary

Architecture:
- Universal Transformer (shared layer across time)
- Single base expert
- LoRA adapters (small, composable)
- Dynamic expert generation per input
- MoE routing over generated experts

Core innovation:
> Replace **static experts** with **dynamic, composable experts**

Expected benefits:
- Extreme parameter efficiency
- Combinatorial capacity
- Input-adaptive computation
- Better scaling under tight constraints

---

## Key Sentence (Pitch)

We replace static experts with dynamically generated experts, allowing model capacity to scale combinatorially while keeping parameters nearly constant.
