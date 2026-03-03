# Process-Reward Guided Surgical Evolution (PRSE)

A test-time scaling algorithm for LLM math reasoning that applies a genetic algorithm to Chain-of-Thought traces, using a step-level Process Reward Model (PRM) to deterministically guide evolutionary operators.

Designed to run in a Google Colab notebook backed by an NVIDIA H100 GPU (80 GB VRAM), high-power CPU, and 180 GB system RAM.

> **Benchmark pivot**: the pipeline now targets the **competition_math (MATH)** dataset, specifically **Level 4 and Level 5** problems where standard Best-of-N fails, allowing PRSE's surgical search to demonstrate compute efficiency.  The earlier GSM8K experiments are preserved in `gsm8k_initial_results.csv`.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [License](#license)

## Overview

PRSE improves LLM math reasoning at inference time by evolving Chain-of-Thought (CoT) traces through a population-based search. Rather than sampling many independent solutions and picking the best one (as in Best-of-N), PRSE iteratively refines traces using two biologically inspired operators:

- **Surgical Mutation**: Identifies the weakest reasoning step (scored by the PRM), truncates the trace at that point, and regenerates from there using a different approach.
- **Logical Grafting (Crossover)**: Combines the validated prefix of one trace with high-scoring insights from another, producing offspring that integrate the strengths of both parents.

A step-level Process Reward Model scores each individual reasoning step, providing fine-grained feedback that guides both operators toward correct solutions.

## Architecture

The codebase is organised into the following components:

### LocalModelManager

Initializes two vLLM pipelines (generator and PRM) and exposes batched generate and score helpers. Models are loaded lazily on first use, allowing the class to be instantiated without a GPU for testing. Defaults are tuned for an H100 80 GB (`max_model_len=8192`, generator 55 %, PRM 35 % of VRAM).

### ReasoningTrace

Immutable dataclass representing one individual in the evolutionary population. Stores the problem statement, its CoT steps, per-step PRM scores, aggregate fitness, and a terminal flag indicating whether the trace contains a boxed final answer.

### ComputeTracker

Thread-safe accounting of prompt tokens, completion tokens, PRM tokens, wall-clock GPU latency, peak VRAM, number of mutations applied, crossovers applied, and total steps truncated. Provides summary dictionaries suitable for logging or Pandas DataFrames.

### PRSEAlgorithm

Orchestrates the evolutionary loop:

1. Initialize the population via batched zero-shot CoT generation.
2. Evaluate every step with the PRM.
3. Early-stop if a perfect terminal trace is found.
4. Apply Surgical Mutation (truncate at the weakest step, regenerate).
5. Apply Logical Grafting/Crossover (splice a high-scoring suffix from one parent onto the validated prefix of another).
6. Repeat for a fixed number of generations or until the budget is exhausted.

### BestOfNRunner

Baseline implementation that generates N independent traces and returns the one with the highest PRM fitness score.

### load_math_problems

Loads Level 4 and Level 5 problems from the `lighteval/MATH` HuggingFace dataset with optional subject filtering and reproducible subsampling.

### setup_output_dirs

Mounts Google Drive automatically when running in Colab and creates the output directory tree under `/content/drive/MyDrive/PRSE_MATH_Results/`.  Falls back to `./prse_math_results/` outside Colab.

### generate_plots

Academic-quality visualisation suite (matplotlib + seaborn, `whitegrid` style) that produces five graphs:

1. **Accuracy vs. Total Compute (Tokens)** – hero graph for the paper.
2. **Accuracy vs. Wall-Clock Time** – latency trade-off.
3. **Compute Breakdown** – stacked bar: generator tokens vs. PRM tokens (PRSE).
4. **Difficulty Degradation** – accuracy grouped by Level 4 / Level 5 for both methods.
5. **PRM Rejection Rate** – histogram of steps truncated per problem.

## Requirements

- Python 3.9 or later
- NVIDIA GPU with CUDA support (H100 80 GB recommended)
- [vLLM](https://github.com/vllm-project/vllm) for high-throughput inference
- [PyTorch](https://pytorch.org/) with CUDA
- [datasets](https://huggingface.co/docs/datasets) for loading the MATH benchmark
- [pandas](https://pandas.pydata.org/) (optional, for tabular output)
- [matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/) (optional, for plots)

## Installation

Clone the repository:

```bash
git clone https://github.com/itsloganmann/Process-Reward-Guided-Surgical-Evolution.git
cd Process-Reward-Guided-Surgical-Evolution
```

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install vllm torch
```

## Usage

Run the main script to execute PRSE and Best-of-N on 50 Level 4/5 MATH problems:

```bash
python prse.py
```

Or inside a Colab or Jupyter notebook:

```python
%run prse.py
```

The script will:
1. Mount Google Drive (in Colab) and create the output directory.
2. Load problems from `lighteval/MATH` (Level 4 & 5).
3. Run PRSE and Best-of-N on each problem, saving results incrementally.
4. Generate all visualisation plots at the end.

### Using Components Individually

```python
from prse import LocalModelManager, PRSEAlgorithm, ComputeTracker

manager = LocalModelManager(
    generator_model_id="Qwen/Qwen2.5-Math-7B-Instruct",
    prm_model_id=None,  # Use heuristic PRM, or specify a real PRM model ID
)

solver = PRSEAlgorithm(
    model_manager=manager,
    population_size=16,
    max_generations=5,
)

best_trace, tracker = solver.solve("Find all real solutions to x^4 - 5x^2 + 4 = 0.")
print(best_trace.as_text())
print(tracker.summary())
```

## Algorithm Details

### Initialization

The population is seeded by generating `population_size` independent CoT traces from the same zero-shot prompt. Each trace is parsed into discrete numbered steps.

### Evaluation

Every step in every trace is scored by the PRM on a 0.0 to 1.0 scale. The aggregate fitness of a trace is the harmonic mean of its step scores. A trace is marked as terminal if its final step contains a boxed answer.

### Selection

Traces are sorted by fitness. The top fraction (controlled by `elite_fraction`) is preserved unchanged. The remaining slots are filled by offspring from mutation and crossover.

### Surgical Mutation

For each selected parent:

1. Find the first step whose PRM score falls below the mutation threshold.
2. Truncate the trace to the validated prefix (all steps before the weak one).
3. Prompt the generator to continue from the truncation point using a different approach.
4. Parse and score the new continuation.

### Logical Grafting (Crossover)

For a selected pair of parents (A and B):

1. Identify Parent A's validated prefix (steps where the score meets or exceeds the threshold).
2. Identify Parent B's high-scoring later steps.
3. Prompt the generator to combine the prefix of A with the insights from B.
4. Parse and score the grafted offspring.

### Termination

The search stops when any of the following conditions are met:

- A terminal trace with fitness at or above the `perfect_fitness_threshold` is found.
- The maximum number of generations is reached.

## Output Files

All output is written under the base directory (Google Drive in Colab, `./prse_math_results/` elsewhere):

| File | Description |
|---|---|
| `prse_vs_bon_telemetry.csv` | Per-problem breakdown: tokens, wall-clock time, accuracy, mutation counts. |
| `raw_evolutionary_traces.jsonl` | Full reasoning traces with step-level PRM scores for qualitative analysis. |
| `summary_metrics.json` | Aggregate statistics (accuracy, total tokens, total time). |
| `plots/accuracy_vs_tokens.{png,pdf}` | Hero graph: accuracy vs. compute. |
| `plots/accuracy_vs_time.{png,pdf}` | Accuracy vs. wall-clock time. |
| `plots/compute_breakdown.{png,pdf}` | Stacked generator / PRM token bar chart. |
| `plots/difficulty_degradation.{png,pdf}` | Accuracy by Level 4 / Level 5. |
| `plots/prm_rejection_rate.{png,pdf}` | Histogram of steps truncated per problem. |

## Configuration

Key parameters for `PRSEAlgorithm`:

| Parameter | Default | Description |
|---|---|---|
| `population_size` | 16 | Number of initial traces to generate |
| `max_generations` | 5 | Maximum evolutionary generations |
| `mutation_threshold` | 0.6 | PRM score below which a step is considered weak |
| `perfect_fitness_threshold` | 0.95 | Fitness at which a terminal trace triggers early stop |
| `max_new_tokens` | 2048 | Token budget per generation call |
| `temperature` | 0.8 | Sampling temperature for the generator |
| `mutation_candidates_per_trace` | 2 | Mutant offspring per selected parent |
| `crossover_candidates` | 2 | Crossover offspring per parent pair |
| `elite_fraction` | 0.25 | Fraction of population preserved between generations |

Key parameters for `LocalModelManager`:

| Parameter | Default | Description |
|---|---|---|
| `generator_model_id` | `Qwen/Qwen2.5-Math-7B-Instruct` | HuggingFace model ID for the generator |
| `prm_model_id` | `peiyi9979/math-shepherd-mistral-7b-prm` | HuggingFace model ID for the PRM (set to `None` for heuristic) |
| `tensor_parallel_size` | 1 | Number of GPUs to shard each model across |
| `max_model_len` | 8192 | Maximum context length for vLLM (H100-optimised) |
| `generator_gpu_memory_utilisation` | 0.55 | Fraction of GPU memory for the generator |
| `prm_gpu_memory_utilisation` | 0.35 | Fraction of GPU memory for the PRM |

## License

This project is provided as-is for research and educational purposes.
