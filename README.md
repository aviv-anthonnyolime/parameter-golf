<img width="3840" height="1280" alt="1920x640-discord" src="https://github.com/user-attachments/assets/90607b26-171f-476a-90ae-69b9dbb7cb30" />

<br>
<br>

**OpenAI Model Craft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).

This challenge is heavily inspired by the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where participants compete to train a model that reaches 3.28 FineWeb validation loss as quickly as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people toward unique architectures (test-time compute, aggressive parameter tying, depth recurrence, low-rank training, ...), compression schemes (low precision, QAT, bitnets, novel tokenizers, ...), and other creative submissions (test-time training, long context, megakernels ...).

If you're familiar with [neural scaling laws](https://arxiv.org/abs/2001.08361), you can consider this challenge a form of L(N) optimization, where the objective is to optimize the lowest loss given a fixed number of parameters (N) unconstrained by data, compute, steps, or architecture. Challenges like the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), which optimizes for a form of L(T) (~lowest time given constrained loss) or the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun), which optimizes for L(D) (lowest loss given constrained dataset size), can be thought of as equivalent challenges in this family.

Ideally, we'd allow for submissions to use arbitrary computational resources. But in order to make the challenge not inaccessibly expensive, we're limiting *leaderboard submissions* to 10 minutes on 8xH100s. However, we'd still love to see submissions that don't meet the compute limitation requirements in our 'Non-record Submissions' section: We're excited to see people push the infinite frontier of parameter limited performance as well.

We also know compute is expensive, so **OpenAI is sponsoring $1,000,000 in compute credits** to help people get started training their models. To request a compute grant, use this form: [Request a Compute Grant](https://openai.com/index/parameter-golf/#credit-form).
When requesting compute, please make sure you choose the appropriate level, write sufficient justification, and **submit with an email tied to a OpenAI / ChatGPT account**.

## Participant Form

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). It helps us attribute challenge submissions and reach out about opportunities with OpenAI. *Completing the form is not required to participate.*

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The Model Craft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.

In June, we plan to hire a small cohort of early-career researchers, targeting current undergraduate students and recent graduates, including Olympiad medalists and elite competitors. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiters.

The challenge runs from March 18th to April 30th.

Happy training!

## Leaderboard

| Run | Score | Author | Summary | Date | Info |
|-----|------:|--------|---------|------|------|
| LeakyReLU² + Legal Score-First TTT + Parallel Muon | 1.1194 | sanjeevmadhav | On PR #549: LeakyReLU(0.5)^2 + TTT + Parallel Muon on the PR #414 stack | 2026-03-23 | [info](records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md) |
| 11L EMA + GPTQ-lite + warmdown3500 | 1.1228 | signalrush | On PR #374: GPTQ-lite clip search + EMA, plus warmdown3500 and QAT@0.15 | 2026-03-22 | [info](records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md) |
| 11L Partial RoPE + LN Scale + EMA + XSA4 | 1.1248 | jfprincz | On PR #287: Partial RoPE (16/64) + layerwise LN scale | 2026-03-21 | [info](records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md) |
| 11L XSA4 + EMA + Int6 MLP3x | 1.1271 | jfprincz | On PR #198: XSA on the last 4 layers + EMA replacing SWA | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md) |
| 11L Efficient Partial XSA | 1.1307 | unnir | On PR #198: Efficient Partial XSA on the deepest 3 layers | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md) |
| 10L Int5-MLP + BigramHash(10240) | 1.1428 | thwu1 | 10 layers, mixed int5/int6 quantization, BigramHash(10240), SWA(0.4), WD=0.04 | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md) |
| Int6 MLP3x + SmearGate + BigramHash | 1.1458 | Raahil Shah | 3x MLP + SmearGate + BigramHash + OrthoInit + Muon WD + SWA | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md) |
| 11L MLP3x + Int6 QAT | 1.1502 | aruniyer | 11 layers, 3x MLP, int6 QAT, zstd-22, WD=0.04, sliding eval | 2026-03-20 | [info](records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md) |
| SmearGate + OrthoInit + Muon WD | 1.1556 | aquariouseworkman | SmearGate + BigramHash + 3x MLP + int6 STE QAT + sliding eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/README.md) |
| 10L Int6 QAT + Zstd MLP2.6x | 1.1586 | yahya010 | 10 layers, int6 QAT + zstd-22, MLP 1344, Muon 0.99, sliding eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/README.md) |
| Mixed Quant + Sliding Window Eval | 1.1630 | aquariouseworkman | Int6 block weights + int8 embeddings + 3x MLP + sliding eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md) |
| Muon WD + 10 layer | 1.1748 | notapplica | Includes prev. wins + Spectral embed init + resid mix | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md) |
| Sliding Window Eval | 1.1925 | Matthew Li | Sliding window evaluation at stride=64, increasing context for eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md) |
| Lora TTT | 1.1928 | samacqua | Test-time training with LORAs | 2026-03-19 | [info](records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md) |
| 4k seq length| 1.2014 | Spokane Way | 4k seq length + better hypers | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md) |
| 2048 seq length | 1.206 | Spokane Way | 2048 seq length (train + val) | 2026-03-18 | [info](records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md) |
| int6 mixed precision | 1.2147 | Nan Liu | 10 layers, mixed int8/int6 | 2026-03-18 | [info](records/track_10min_16mb/2026-03-19_10L_MixedPrecision/README.md) |
| fp16 Embed | 1.2197 | Renier Velazco | FP16 Tied Embedding + LR/Warmdown Tuning | 2026-03-18 | [info](records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md) |
| Naive Baseline | 1.2244 | Baseline | 9layer 512dim 1024vocab TiedEmbeddings 4 KV heads | 2026-03-18 | [info](records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) |

#### Notable Non-Record Runs

| Run | Score | Author | Summary | Date | Info |
|-----|------:|--------|---------|------|------|
| 4-Hour Baseline | 1.2074 | Will DePue | Testing unlimited compute, 4 hours on 8xH100 | 2026-03-18 | [info](records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md) |

## Getting Started

### Training Your First Model (Mac with Apple Silicon)

If you have an Apple laptop or desktop with Apple Silicon, we've set up a simple MLX training script to help you start iterating locally.

If you don't have a Mac with Apple Silicon, you can run an adapted version of this script without MLX support. Just ask [Codex](https://openai.com/codex/) to refactor it; the change is straightforward. It may still be fairly slow, so we recommend jumping straight to cloud GPUs with Runpod.

First, clone the repository, create a fresh Python environment, and install the packages needed for the MLX path plus dataset download:

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

Download our cached version of FineWeb with the 1024-token vocabulary:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default this downloads the full validation split plus 80 training shards (8B tokens). For a smaller local smoke subset, pass `--train-shards 1`, for example `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.

Then run a small MLX training job:

```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_LOG_EVERY=20 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Validation always runs on the full `fineweb_val_*` split, which is the fixed first-50k-document set. The smoke command above skips periodic validation and just prints the final `val_loss` and `val_bpb` once at the end.

### Scaling Up to a Remote Machine

Once you're happy with your local tests, or you want more compute, switch to a remote CUDA machine.

You can rent GPUs from anywhere, but OpenAI is partnering with Runpod to make setup as easy as possible.  

#### Launching a 1xH100 Pod

1. First, [create a Runpod account](https://console.runpod.io/deploy). You should also set up an SSH key in the Settings tab on the left so you can connect to your remote machine. If you're new to this, ask Codex to help you set it up.

2. Once you've set up your account, create a new GPU Cloud Pod. You can choose whichever GPU SKU you'd like. Final leaderboard submissions must run in under 10 minutes on 8xH100s (specifically the SXM variant), but we strongly recommend testing and running experiments on cheaper SKUs first, since an 8xH100 box can cost around $20/hour.

3. Let's start with a 1xH100 pod. Deploy using the official Parameter Golf template: [Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th). Enable SSH terminal access, leaving the other settings at their defaults. Deploy your pod and SSH into it once it's up. You should land in `/workspace/`.

On your remote machine, clone the repo onto local disk. All Python dependencies are already pre-installed in the image.

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

Download our cached version of FineWeb. We'll use the 1024-token vocabulary for now.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This defaults to the full validation split plus 80 training shards (8B tokens). If you only want a smaller subset while iterating, pass `--train-shards N`, for example `--train-shards 1`.

Launch your first training run. Note that we're passing `nproc_per_node=1` because we're running on a single H100 GPU in this case.

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

By default, `train_gpt.py` keeps its ~10 minute wallclock cap. If you want a longer run, override it explicitly, for example `MAX_WALLCLOCK_SECONDS=0`.

By default, this command prints `train_loss` step logs during training and prints `val_loss`, `val_bpb`, and compressed model size in the final `final_int8_zlib_roundtrip` lines at the end. If you want periodic validation logs during the run, set `VAL_LOSS_EVERY`, for example `VAL_LOSS_EVERY=200`. For the baseline config, the final `val_bpb` should land around ~1.2 with a compressed model size under 16MB.

For dataset export, tokenizer export, and docs-cache rebuild instructions, see [data/README.md](data/README.md).

Evaluation will be in the RunPod environment with all packages installed. `requirements.txt` is provided as a reference if you want to self-setup.

#### Setting Up on Amazon Linux 2 (AWS / SageMaker)

If you're running on an Amazon Linux 2 instance (e.g. `g5dn.12xlarge` on SageMaker), several issues need to be addressed:

- **GCC 7** (default) — NumPy 2.x requires GCC >= 9.3. GCC 10 is already installed but under a different binary name.
- **CMake 2.8** (default) — `sentencepiece` and `pyarrow` require CMake >= 3.12.
- **No Go toolchain** — `wandb >= 0.18` builds `wandb-core` from source and requires Go >= 1.26.1.
- **No Rust toolchain** — `wandb >= 0.18` builds `gpu_stats` from source and requires Rust/Cargo.
- **binutils 2.29** (default) — Rust's `ring` crate needs binutils >= 2.30 for AVX2 assembly. The setup script builds a newer version from source.
- **SageMaker volume is only 4.8GB** — PyTorch + CUDA libs are ~2GB+. The venv must live on the root filesystem (`/`, 135GB) not in `/home/ec2-user/SageMaker`.

The setup script handles all of the above automatically with no changes to training code. The first run takes longer (~5-10 extra minutes) while Go, Rust, and binutils are installed; subsequent runs skip already-installed tools.

**First run** (creates venv, symlinks, installs deps):

```bash
bash scripts/setup_aws_al2.sh
```

**After each restart** (re-exports compilers, re-activates venv — skips reinstall):

```bash
source /opt/pg-venv/bin/activate
bash scripts/setup_aws_al2.sh --skip-install
```

**How the data symlinks work:** `data/datasets` and `data/tokenizers` are symlinked to `/opt/pg-data/` on the root filesystem (135GB free). The download script and `train_gpt.py` follow the symlinks transparently — no code changes needed.

```text
data/datasets   →  /opt/pg-data/datasets   (large .bin files, ~20GB)
data/tokenizers →  /opt/pg-data/tokenizers (small tokenizer models)
```

> **Note:** CC/CXX exports only last for the current shell session.

#### Git Push via SSH (Remote Machines)

The queue runner auto-commits and pushes results after each run. GitHub no longer supports password authentication, so you must use SSH keys. Without this, pushes will fail with `Invalid username or token`.

Run the setup script **on your remote machine** (RunPod / AWS):

```bash
bash scripts/setup_git_ssh.sh
```

This will:

1. Generate an ED25519 SSH key (if none exists)
2. Print the public key — copy it and add it at <https://github.com/settings/ssh/new>
3. Switch the `origin` remote from HTTPS to SSH automatically
4. Test the connection

If you've already set up SSH keys on the machine, the script will skip key generation and just fix the remote URL.

Alternatively, if you prefer HTTPS with a token, you can set the remote manually:

```bash
# Using a GitHub Personal Access Token (PAT)
git remote set-url origin https://<YOUR_TOKEN>@github.com/<owner>/parameter-golf.git
```

### Experiment Tooling

The repo includes tooling for tracking experiments, queuing overnight runs, and visualizing results. Install the extra dependencies first:

```bash
pip install pyyaml matplotlib wandb
```

#### Run Naming & Log Files

Each training run automatically gets a **docker-style name** (e.g. `brave-falcon`) derived from the `RUN_ID`. Log files are named with date, time, model params, and docker name for easy sorting:

```text
logs/2026-03-20_143052_10L-1U-512d-2mlp_brave-falcon.txt
```

The same name is used for the W&B run: `brave-falcon_10L-1U-512d-2mlp`.

#### Weights & Biases (W&B) Integration

Enable W&B logging by setting `WANDB_ENABLED=1`. Logs all hyperparameters as config, and tracks `train_loss`, `step_avg_ms`, `lr_scale` every `TRAIN_LOG_EVERY` steps (default: 10), plus `val_loss`/`val_bpb` on every validation.

```bash
# Single run with W&B
WANDB_ENABLED=1 \
RUN_ID=my_experiment \
torchrun --standalone --nproc_per_node=4 experiments/phase1_UT/train_gpt_ut.py
```

Environment variables:

| Variable          | Default                              | Description                                              |
| ----------------- | ------------------------------------ | -------------------------------------------------------- |
| `WANDB_ENABLED`   | `0`                                  | Set to `1` to activate W&B logging                       |
| `WANDB_ENTITY`    | `citaman`                            | W&B team/entity name                                     |
| `WANDB_PROJECT`   | `Openai-challenge-parameter-golf`    | W&B project name                                         |
| `TRAIN_LOG_EVERY` | `10`                                 | Log training metrics every N steps (both console and W&B)|

**API key setup** — the queue runner loads credentials automatically:

- `.env` — committed template, lists all variables with placeholder values (safe to commit)
- `.env.local` — your actual secrets, **gitignored**, never committed

```bash
# One-time setup: copy template and fill in your W&B key
cp .env .env.local
# then edit .env.local and set WANDB_API_KEY=<your key>
```

`.env.local` overrides `.env`. Existing shell env vars are never overwritten, so you can still pass keys inline if needed.

#### Results Tracking (JSONL)

After each run completes, results are automatically appended to two JSONL files:

- **Per-experiment**: `experiments/phase1_UT/results.jsonl`
- **Global**: `results/all_runs.jsonl`

Each entry contains: `run_id`, `docker_name`, `params_tag`, `val_loss`, `val_bpb`, `val_bpb_int8`, `val_bpb_ttt`, `avg_ms_per_step`, `compressed_size_mb`, `peak_memory_mib`, and the full hyperparameters dict.

#### Ranking & Charts

View a ranked leaderboard of all runs:

```bash
# Global leaderboard (all experiments)
python scripts/ranking.py

# Per-experiment leaderboard
python scripts/ranking.py experiments/phase1_UT/results.jsonl

# Generate matplotlib charts (bar chart, loss line, bpb line, size scatter)
python scripts/ranking.py --chart
python scripts/ranking.py --chart --out my_charts/
```

Example output:

```text
Rank  Docker Name          Experiment    val_bpb  bpb_int8  bpb_ttt    loss  ms/step  steps  size_MB             Params
   1  swift-eagle          phase1_UT      1.1800    1.1900   1.1700  1.4000    583.3   1200    14.50   10L-2U-512d-2mlp
   2  calm-otter           phase1_UT      1.2000    1.2100   1.1900  1.4200   1000.0    800    15.07   12L-1U-640d-2mlp
   3  brave-falcon         phase1_UT      1.2200    1.2300   1.2100  1.4500    548.0   1000    13.83   10L-1U-512d-2mlp
```

#### Queue Runner (Overnight Runs)

Define multiple runs in a YAML file and execute them sequentially. After each run, results are committed and pushed to git automatically so you can check progress remotely.

```bash
# Preview what will run (no execution)
python scripts/run_queue.py experiments/phase1_UT/queue.yaml --dry-run

# Run the queue (commits + pushes after each run)
python scripts/run_queue.py experiments/phase1_UT/queue.yaml

# Run without pushing (commit locally only)
python scripts/run_queue.py experiments/phase1_UT/queue.yaml --no-push
```

Git commits follow this format:

```text
[brave-falcon] - [10L-1U-512d-2mlp] - [phase1_UT] - (loss: 1.4452) (val_bpb: 1.2147)
```

**Example queue file** (`experiments/phase1_UT/queue.yaml`):

```yaml
script: experiments/phase1_UT/train_gpt_ut.py
nproc: 4    # 4 for g5dn (4×A10G), 8 for RunPod (8×H100)

defaults:
  DATA_PATH: ./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH: ./data/tokenizers/fineweb_1024_bpe.model
  VOCAB_SIZE: "1024"
  MAX_WALLCLOCK_SECONDS: "2000"
  TRAIN_LOG_EVERY: "10"
  WANDB_ENABLED: "1"

runs:
  - RUN_ID: baseline_10L_512d
    NUM_LAYERS: "10"
    NUM_UNIQUE_LAYERS: "1"
    MODEL_DIM: "512"

  - RUN_ID: test_10L_2U_512d
    NUM_LAYERS: "10"
    NUM_UNIQUE_LAYERS: "2"
    MODEL_DIM: "512"
```

A longer overnight sweep is also provided in `experiments/phase1_UT/queue_overnight.yaml` (6 runs, ~3.5h on 4×A10G) covering depth, unique-layer, and width ablations.

#### Promote Best Runs to RunPod

After local experiments, promote the top candidates to an 8×H100 queue:

```bash
# Pick top 3 runs → generate RunPod queue
python scripts/promote.py --top 3

# From a specific results file
python scripts/promote.py --top 5 --from experiments/phase1_UT/results.jsonl

# Custom settings
python scripts/promote.py --top 3 --nproc 8 --wallclock 600 --out runpod_queue.yaml
```

This generates a ready-to-run YAML queue file with `nproc: 8` and `MAX_WALLCLOCK_SECONDS: 600`, plus prints the `torchrun` commands for manual use.

#### GPU Memory Estimation

Before scaling up model dimensions, check if it fits in your GPU memory:

```bash
# Default config
python scripts/model_size.py experiments/phase1_UT/train_gpt_ut.py

# Custom config
NUM_LAYERS=12 MODEL_DIM=640 python scripts/model_size.py experiments/phase1_UT/train_gpt_ut.py

# Custom GPU budget
NPROC=8 GPU_MEM_GB=80 python scripts/model_size.py experiments/phase1_UT/train_gpt_ut.py
```

#### Killing a Training Run

If you need to stop a training run (stuck process, wrong config, etc.), use the kill script:

```bash
bash scripts/kill_training.sh
```

This will:

1. Gracefully kill processes in dependency order: `run_queue.py` → `torchrun` → `torch.distributed` → `train_gpt`
2. Wait 2 seconds, then force-kill (`SIGKILL`) anything still alive
3. Clean up orphaned NCCL shared memory (`/dev/shm/nccl-*`)
4. Print current GPU status so you can confirm resources are freed

Safe to run multiple times — won't error if nothing is running.

#### Scripts Summary

| Script                     | Description                                                       |
|----------------------------|-------------------------------------------------------------------|
| `scripts/ranking.py`       | Display ranked leaderboard + matplotlib charts from JSONL results |
| `scripts/run_queue.py`     | Run a YAML queue of experiments with auto git commit/push         |
| `scripts/promote.py`       | Pick best local runs and generate RunPod 8×H100 queue             |
| `scripts/model_size.py`    | Estimate compressed model size + GPU memory for DDP training      |
| `scripts/naming.py`        | Docker-style adjective-animal name generator (used internally)    |
| `scripts/kill_training.sh` | Kill all training processes and free GPU resources                |
| `scripts/setup_git_ssh.sh` | Set up SSH-based git push on remote machines                      |

## FAQ

**What exactly counts toward the 16MB artifact size?**

The submission artifact is computed as code bytes plus compressed model bytes. All counted code should live in the `train_gpt.py` script.
The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes.
No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues reproducing submissions should be raised on the PR. If you find an issue with a record on the leaderboard or find a record isn't reproducible, please let us know and add an Github Issue describing your findings.

**What counts as 'external compute'? For example, is it fair to tune my hyperparameters offline?**

There's no perfectly clear answer here and it's hard to draw a clean line around what does or does not count as external compute. For now, we're reserving the right to disqualify runs that are not in the spirit of the challenge. Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence that you're sneaking in additional compute unfairly, such as brute-forcing ridiculous seeds, we won't allow it. Use your best judgment and there's no penalty for asking questions.

**What are the restrictions on evaluation?**

We won't accept submissions that take more than 10 minutes on 8xH100 to evaluate (Note: This limit is in addition to the 10 minutes of training time allowed!), but otherwise you're free to evaluate however. As with modded-nanogpt, we allow evaluation at any sequence length. And, obviously, you aren't allowed to access any training data during evaluation, unless you pay for those bits in the <16MB limit. We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods. You CANNOT access validation data during training, e.g. by compressing it into your 16mb with "paid prefix".

If it isn't abundantly obvious: You can't cheat on your test loss. You can't cheat by training on the validation set before you evaluate on the validation set. The validation language around test-time training has been confusing people: you are only allowed to test-time train on validation set tokens *you've already evaluated your model on*, since those tokens have already been graded!

**What is the process for accepting new submissions?**

Since all submissions are public, we're accepting record submissions chronologically depending on their PR creation time. The leaderboard may take time to update due to verification and review of submissions, so pay consideration to what the current SOTA PR is when submitting. As explained below, submissions should exceed the SOTA record with sufficient statistical significance in order to be accepted for the leaderboard. Otherwise, submissions may be accepted as 'non-record submissions' given they are sufficiently unique or interesting.

**Can I import XYZ package or library?**

Yes, you're free to import any package or library you want, so long as it does not unjustly violate the rules on evaluation, compute, training time, code size or otherwise. Just include a requirements.txt in your records folder and mention setup instructions in your README.md. Since you don't pay for bits imported in Python libraries, limitations clearly apply: You can't sneak in extra compute, capabilities, or massively increase effective code size with custom libraries, but importing FlashAttention, etc. is completely fine.

## Submission Process

New SOTA records must fulfill the following criteria:

1. They must beat the existing SOTA by at least 0.005 nats. As in modded-nanogpt, because of inter-run variance all submissions must provide enough run logs to show at `p < 0.01` that they achieved the required 0.005-nat improvement. For submissions that improve speed through systems optimization without changing the ML, this requirement is waived.

2. If changes are made to the tokenizer or dataset, prove with certainty that the val_bpb is correctly calculated. Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score.

3. Reproducibly run in under 10 minutes on 8xH100s.

All submissions should be made as a pull request that only adds a new folder to the appropriate `/records` subfolder and includes the following files. Submissions without the full set of requirements will not be accepted.

1. A README.md file that explains the submission in reasonable detail.

2. A `submission.json` file (see the example runs) that includes your name, GitHub ID, `val_bpb`, and related metadata.

3. A train log, automatically produced by your script. Please demonstrate a statistically significant win. Most often, submitting an average over 3 training runs is sufficient.

4. A `train_gpt.py` script and any other dependencies. Note: this must successfully compile and run within the records folder. Broken scripts will not be accepted.

### Non-record Submissions

Submissions are also open to unique and interesting approaches that might not beat the existing SOTA, but still satisfy the 16MB artifact limit. We strongly encourage participants to submit implementations for weird or out-of-the-box ideas, in-progress or unoptimized solutions, so long as they run successfully, or even interesting negative results. We're excited to see what you come up with. We'll still maintain a high bar for non-record submissions, so be sure to justify your ideas and results in detail when submitting.

We also accept non-record submissions to an unlimited compute track for runs that are not intended to meet the 10-minute cutoff. Just note as such in your README file.

Non-record submissions should be made in the same fashion as SOTA records, as described above.

#### PRs on Core Code

The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but the best models should stay in the `/records` folder.

## Support

Join the [OpenAI Discord server](https://discord.com/invite/openai) and visit the Parameter Golf channels (#parameter-golf-discussions, #parameter-golf-announcements) and ask questions.

This repository adapts code from `modded-nanogpt`, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
