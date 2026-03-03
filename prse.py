"""
Process-Reward Guided Surgical Evolution (PRSE)
================================================
A novel test-time scaling algorithm for LLM math reasoning that applies a
Genetic Algorithm to Chain-of-Thought traces, using a step-level Process
Reward Model (PRM) to deterministically guide evolutionary operators.

Designed to run in a Google Colab notebook backed by an NVIDIA H100 GPU
(80 GB VRAM), high-power CPU, and 180 GB System RAM.

Architecture overview
---------------------
LocalModelManager
    Initialises two vLLM pipelines (generator + PRM) and exposes batched
    generate / score helpers.  Optimised for the H100 80 GB by default.

ReasoningTrace
    Immutable-ish dataclass representing one individual in the population.
    Stores the problem, its CoT steps, per-step PRM scores, aggregate
    fitness, and a terminal flag.

ComputeTracker
    Thread-safe accounting of prompt tokens, completion tokens, PRM tokens,
    wall-clock GPU latency, peak VRAM, mutation counts, and steps truncated.

PRSEAlgorithm
    Orchestrates the evolutionary loop:
      1. Initialise population via batched zero-shot CoT generation.
      2. Evaluate every step with the PRM.
      3. Early-stop if a perfect terminal trace is found.
      4. Apply Surgical Mutation (truncate at weakest step, regenerate).
      5. Apply Logical Grafting / Crossover (splice high-scoring suffix
         from parent B onto the validated prefix of parent A).
      6. Repeat for a fixed number of generations or until budget exhausted.

BestOfNRunner
    Baseline: generate N independent traces, pick the highest-fitness one.

load_math_problems()
    Load Level 4 and Level 5 problems from the hendrycks/competition_math dataset
    (HuggingFace), which provides the challenging benchmark where BoN fails.

setup_output_dirs()
    Mount Google Drive (when running in Colab) and create the results
    directory tree under /content/drive/MyDrive/PRSE_MATH_Results/.

append_telemetry_row() / append_trace_record()
    Incremental per-problem persistence helpers that write to CSV and JSONL
    after every single problem so the run survives Colab timeouts.

generate_plots()
    Academic-quality visualisation suite (matplotlib + seaborn) producing
    five high-resolution PNG/PDF graphs for the paper.

main()
    Loads Level 4/5 MATH problems, runs PRSE and Best-of-N, saves results
    incrementally, and generates all visualisation plots at the end.

Usage
-----
    python prse.py

    # Or inside a Colab / Jupyter cell:
    # %run prse.py
"""

from __future__ import annotations

import os

# Must be set before any protobuf-backed library (e.g. datasets / tensorboard)
# is imported.  Protobuf ≥ 4.x removed MessageFactory.GetPrototype; the
# pure-Python implementation restores the legacy API.  To avoid this overhead
# in production, pin protobuf to a 3.x release (e.g. protobuf==3.20.*).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import csv
import dataclasses
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level logger.  Downstream code should use ``logging.getLogger`` with
# its own ``__name__`` rather than printing directly.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily so the module itself can be
# imported and unit-tested without a GPU present.
# ---------------------------------------------------------------------------
try:
    from vllm import LLM, SamplingParams  # type: ignore
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from datasets import load_dataset  # type: ignore
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")  # non-interactive backend for server/Colab use
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# ===========================================================================
# Section 1 – Data structures
# ===========================================================================


@dataclasses.dataclass
class ReasoningTrace:
    """One individual in the evolutionary population.

    Parameters
    ----------
    problem_statement:
        The original math problem text.
    steps:
        Ordered list of reasoning steps parsed from the model output.
        Each step is a single string (may contain LaTeX / arithmetic).
    step_scores:
        Per-step confidence scores in [0.0, 1.0] assigned by the PRM.
        ``len(step_scores) == len(steps)`` after evaluation.
    total_fitness:
        Aggregate fitness derived from ``step_scores`` (e.g. mean or
        product).  Defaults to 0.0 before evaluation.
    is_terminal:
        ``True`` when the final step contains a boxed / \boxed{} answer
        that the PRM has rated as a valid solution step.
    """

    problem_statement: str
    steps: List[str]
    step_scores: List[float] = dataclasses.field(default_factory=list)
    total_fitness: float = 0.0
    is_terminal: bool = False

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def prefix(self, k: int) -> List[str]:
        """Return the first *k* steps (safe slice)."""
        return self.steps[:k]

    def as_text(self, step_sep: str = "\n") -> str:
        """Reconstruct the full CoT trace as a single string."""
        return step_sep.join(self.steps)

    def prefix_as_text(self, k: int, step_sep: str = "\n") -> str:
        """Reconstruct only the first *k* steps."""
        return step_sep.join(self.prefix(k))

    def first_weak_step(self, threshold: float) -> Optional[int]:
        """Return the index of the first step whose score < *threshold*.

        Returns ``None`` when all steps are above the threshold, meaning
        the trace is already strong end-to-end.
        """
        for i, score in enumerate(self.step_scores):
            if score < threshold:
                return i
        return None

    def clone_with(self, **kwargs) -> "ReasoningTrace":
        """Return a shallow copy with selected fields overridden."""
        d = dataclasses.asdict(self)
        d.update(kwargs)
        return ReasoningTrace(**d)


# ===========================================================================
# Section 2 – Compute tracking
# ===========================================================================


class ComputeTracker:
    """Thread-safe accounting of computation consumed during a search run.

    Tracked quantities
    ------------------
    prompt_tokens     : Total tokens in all prompts sent to the generator.
    completion_tokens : Total tokens generated by the generator.
    prm_tokens        : Total tokens processed by the PRM (prompt + output).
    gpu_latency_s     : Cumulative wall-clock time (seconds) spent waiting
                        for GPU inference.
    peak_vram_mb      : Peak GPU VRAM consumed (MB), sampled via PyTorch.
    mutations_applied : Number of Surgical Mutation offspring generated.
    crossovers_applied: Number of Logical Grafting offspring generated.
    steps_truncated   : Total reasoning steps truncated by Surgical Mutation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.prm_tokens: int = 0
        self.gpu_latency_s: float = 0.0
        self.peak_vram_mb: float = 0.0
        self.mutations_applied: int = 0
        self.crossovers_applied: int = 0
        self.steps_truncated: int = 0

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def add_generation(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float,
    ) -> None:
        """Record a batched generation call."""
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.gpu_latency_s += latency_s
            self._refresh_vram()

    def add_prm_call(self, tokens: int, latency_s: float) -> None:
        """Record a PRM scoring call."""
        with self._lock:
            self.prm_tokens += tokens
            self.gpu_latency_s += latency_s
            self._refresh_vram()

    def record_mutation(self, steps_truncated: int) -> None:
        """Record one Surgical Mutation event.

        Parameters
        ----------
        steps_truncated:
            Number of steps that were dropped from the parent trace before
            the mutation continuation was generated.
        """
        with self._lock:
            self.mutations_applied += 1
            self.steps_truncated += steps_truncated

    def record_crossover(self) -> None:
        """Record one Logical Grafting (crossover) event."""
        with self._lock:
            self.crossovers_applied += 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.prm_tokens

    def summary(self) -> Dict[str, Any]:
        """Return a plain-dict snapshot suitable for Pandas / logging."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prm_tokens": self.prm_tokens,
            "total_tokens": self.total_tokens,
            "gpu_latency_s": round(self.gpu_latency_s, 3),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "mutations_applied": self.mutations_applied,
            "crossovers_applied": self.crossovers_applied,
            "steps_truncated": self.steps_truncated,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_vram(self) -> None:
        """Sample current VRAM and update the peak (call under lock)."""
        if not TORCH_AVAILABLE:
            return
        try:
            if torch.cuda.is_available():
                used = torch.cuda.max_memory_allocated() / (1024 ** 2)
                if used > self.peak_vram_mb:
                    self.peak_vram_mb = used
        except Exception:
            pass


# ===========================================================================
# Section 3 – Local model manager
# ===========================================================================


class LocalModelManager:
    """Manages vLLM inference pipelines for the generator and PRM.

    Parameters
    ----------
    generator_model_id:
        HuggingFace model identifier for the instruction-tuned math
        generator (e.g. ``"Qwen/Qwen2.5-Math-7B-Instruct"``).
    prm_model_id:
        HuggingFace model identifier for the process reward model
        (e.g. ``"peiyi9979/math-shepherd-mistral-7b-prm"``).
        When set to ``None`` a lightweight heuristic PRM is used instead
        (useful for testing without a second GPU model loaded).
    tensor_parallel_size:
        Number of GPUs to shard each model across.  Set to 1 for a single
        H100.
    max_model_len:
        Maximum context length passed to vLLM.  8192 is a good default for
        the H100 80 GB to accommodate long MATH reasoning traces.
    generator_gpu_memory_utilisation:
        Fraction of GPU memory reserved for the generator.  0.55 leaves
        enough headroom for the PRM on the same 80 GB H100.
    prm_gpu_memory_utilisation:
        Fraction of GPU memory reserved for the PRM.  If both models share
        the same GPU, the two fractions should sum to ≤ 0.95.

    Notes
    -----
    vLLM is loaded lazily the first time ``generate`` or ``score_steps``
    is called so that the class can be instantiated in environments
    without a GPU for unit-testing purposes.
    """

    def __init__(
        self,
        generator_model_id: str = "Qwen/Qwen2.5-Math-7B-Instruct",
        prm_model_id: Optional[str] = "peiyi9979/math-shepherd-mistral-7b-prm",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        generator_gpu_memory_utilisation: float = 0.55,
        prm_gpu_memory_utilisation: float = 0.35,
    ) -> None:
        self.generator_model_id = generator_model_id
        self.prm_model_id = prm_model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.generator_gpu_memory_utilisation = generator_gpu_memory_utilisation
        self.prm_gpu_memory_utilisation = prm_gpu_memory_utilisation

        self._generator: Optional["LLM"] = None  # noqa: F821
        self._prm_llm: Optional["LLM"] = None  # noqa: F821
        self._log = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_generator(self) -> None:
        """Initialise the generator vLLM engine if not already done."""
        if self._generator is not None:
            return
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed.  Run: pip install vllm"
            )
        self._log.info(
            "Loading generator model '%s' (tensor_parallel=%d, "
            "max_model_len=%d, gpu_mem_util=%.2f)",
            self.generator_model_id,
            self.tensor_parallel_size,
            self.max_model_len,
            self.generator_gpu_memory_utilisation,
        )
        self._generator = LLM(
            model=self.generator_model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.generator_gpu_memory_utilisation,
            trust_remote_code=True,
        )
        self._log.info("Generator ready.")

    def _ensure_prm(self) -> None:
        """Initialise the PRM vLLM engine if not already done."""
        if self._prm_llm is not None or self.prm_model_id is None:
            return
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed.  Run: pip install vllm"
            )
        self._log.info(
            "Loading PRM model '%s' (gpu_mem_util=%.2f)",
            self.prm_model_id,
            self.prm_gpu_memory_utilisation,
        )
        self._prm_llm = LLM(
            model=self.prm_model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.prm_gpu_memory_utilisation,
            trust_remote_code=True,
        )
        self._log.info("PRM ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        n: int = 1,
    ) -> Tuple[List[str], int, int]:
        """Batched generation via vLLM.

        Parameters
        ----------
        prompts:
            List of prompt strings; each element is one independent
            generation request.
        max_new_tokens:
            Maximum number of tokens to generate per prompt.
        temperature:
            Sampling temperature.
        top_p:
            Nucleus sampling probability.
        n:
            Number of completions per prompt (only the first is returned
            when ``n == 1``).

        Returns
        -------
        texts:
            Flat list of generated strings, one per (prompt, n-sample)
            pair – i.e. length ``len(prompts) * n``.
        prompt_tokens:
            Approximate total prompt tokens processed (sum over batch).
        completion_tokens:
            Total completion tokens generated (sum over batch).
        """
        self._ensure_generator()
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=n,
        )
        outputs = self._generator.generate(prompts, sampling_params)
        texts: List[str] = []
        prompt_tok = 0
        completion_tok = 0
        for req in outputs:
            prompt_tok += len(req.prompt_token_ids)
            for out in req.outputs:
                texts.append(out.text)
                completion_tok += len(out.token_ids)
        return texts, prompt_tok, completion_tok

    def score_steps(
        self,
        problem: str,
        steps: List[str],
    ) -> Tuple[List[float], int]:
        """Score each step in a reasoning trace using the PRM.

        The PRM is queried once per step: the prompt consists of the
        problem statement followed by all steps up to and including the
        current one, and the model is asked to rate the current step on a
        scale from 0 (incorrect / irrelevant) to 1 (correct and useful).

        If no PRM model is configured (``prm_model_id is None``), a
        heuristic fallback is used (see ``_heuristic_score``).

        Parameters
        ----------
        problem:
            The original problem statement.
        steps:
            The individual reasoning steps to score.

        Returns
        -------
        scores:
            List of floats in [0.0, 1.0], one per step.
        total_tokens:
            Approximate number of tokens consumed by all PRM calls.
        """
        if not steps:
            return [], 0

        if self.prm_model_id is None:
            scores = [_heuristic_score(step) for step in steps]
            return scores, 0

        self._ensure_prm()
        prompts = []
        for i, step in enumerate(steps):
            context = "\n".join(steps[: i + 1])
            prompt = _build_prm_prompt(problem, context, step)
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8,  # Only need a short numeric response.
        )
        outputs = self._prm_llm.generate(prompts, sampling_params)

        scores: List[float] = []
        total_tokens = 0
        for req in outputs:
            total_tokens += len(req.prompt_token_ids)
            raw = req.outputs[0].text.strip()
            total_tokens += len(req.outputs[0].token_ids)
            scores.append(_parse_prm_output(raw))
        return scores, total_tokens


# ===========================================================================
# Section 4 – Prompt templates
# ===========================================================================


def _build_zero_shot_cot_prompt(problem: str) -> str:
    """Construct a zero-shot Chain-of-Thought prompt for the generator.

    The prompt instructs the model to reason step-by-step and to present
    each step on its own numbered line, finishing with ``\boxed{answer}``.
    """
    return (
        "Solve the following math problem step by step.\n"
        "Show each reasoning step on a new line, prefixed with 'Step N:' "
        "(where N is the step number).\n"
        "Conclude with: 'The answer is \\boxed{<answer>}'\n\n"
        f"Problem: {problem}\n\n"
        "Solution:"
    )


def _build_mutation_prompt(problem: str, prefix_text: str, step_index: int) -> str:
    """Construct a prompt for the Surgical Mutation operator.

    The generator is given the valid prefix (steps 1 … k-1) and asked to
    produce a *different* continuation starting from step k.  This is the
    core of the Surgical Mutation operator: instead of mutating the whole
    trace, only the first weak point is targeted.

    Parameters
    ----------
    problem:
        The original problem statement.
    prefix_text:
        The validated prefix steps joined as a single string.
    step_index:
        The 1-based step number at which the new branch should start.
    """
    return (
        "You are given a math problem and the beginning of a correct solution.\n"
        "Continue the solution from the indicated step using a DIFFERENT "
        "mathematical approach or technique than what was tried before.\n\n"
        f"Problem: {problem}\n\n"
        f"Validated prefix (Steps 1–{step_index - 1}):\n{prefix_text}\n\n"
        f"Now write a NEW Step {step_index} and continue to the final answer.\n"
        "Conclude with: 'The answer is \\boxed{<answer>}'\n\n"
        f"Step {step_index}:"
    )


def _build_crossover_prompt(
    problem: str,
    prefix_a: str,
    insights_b: str,
    next_step_index: int,
) -> str:
    """Construct a prompt for the Logical Grafting (crossover) operator.

    Parent A's validated prefix is combined with the high-scoring later
    insights from Parent B.  The generator is asked to integrate Parent
    B's mathematical insights to complete Parent A's solution.

    Parameters
    ----------
    problem:
        The original problem statement.
    prefix_a:
        Parent A's validated prefix as text.
    insights_b:
        Parent B's high-scoring steps as text – the "genetic material"
        being grafted.
    next_step_index:
        The step number at which grafting begins (len(prefix_a_steps) + 1).
    """
    return (
        "You are solving a math problem.  You have a correct solution prefix "
        "from one attempt (Parent A) and some useful later insights from "
        "another attempt (Parent B).  Combine them to produce a complete, "
        "correct solution.\n\n"
        f"Problem: {problem}\n\n"
        f"Parent A's validated prefix (Steps 1–{next_step_index - 1}):\n"
        f"{prefix_a}\n\n"
        "Parent B's mathematical insights (to be integrated):\n"
        f"{insights_b}\n\n"
        f"Now write Step {next_step_index} and complete the solution, "
        "integrating the insights from Parent B where useful.\n"
        "Conclude with: 'The answer is \\boxed{<answer>}'\n\n"
        f"Step {next_step_index}:"
    )


def _build_prm_prompt(problem: str, context: str, current_step: str) -> str:
    """Construct a prompt that asks the PRM to score a single step.

    The PRM is instructed to return a single floating-point number between
    0.0 and 1.0 representing the correctness and logical soundness of the
    current step given the preceding context.
    """
    return (
        "You are a mathematical reasoning evaluator.  Given a problem and "
        "the reasoning steps taken so far, rate the quality of the LAST step "
        "on a scale from 0.0 (completely wrong or irrelevant) to 1.0 "
        "(perfectly correct and logically sound).\n\n"
        f"Problem: {problem}\n\n"
        f"Reasoning so far:\n{context}\n\n"
        "Return ONLY a single decimal number between 0.0 and 1.0.\n"
        "Score:"
    )


# ===========================================================================
# Section 5 – Helper utilities
# ===========================================================================


def _parse_steps(raw_text: str) -> List[str]:
    """Parse a model-generated CoT trace into individual steps.

    Tries to split on numbered step markers (``Step N:``).  Falls back to
    splitting on double-newlines and then single newlines.

    Parameters
    ----------
    raw_text:
        The raw model output to parse.

    Returns
    -------
    A non-empty list of step strings.  If parsing produces no steps the
    entire output is returned as a single step.
    """
    # Attempt 1: explicit "Step N:" markers.
    pattern = r"(?:Step\s+\d+\s*:)"
    parts = re.split(pattern, raw_text, flags=re.IGNORECASE)
    steps = [p.strip() for p in parts if p.strip()]
    if len(steps) >= 2:
        return steps

    # Attempt 2: paragraph breaks.
    parts = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        return parts

    # Attempt 3: single newlines.
    parts = [p.strip() for p in raw_text.splitlines() if p.strip()]
    if parts:
        return parts

    # Fallback: whole text is one step.
    return [raw_text.strip()] if raw_text.strip() else ["(empty)"]


def _parse_prm_output(raw: str) -> float:
    """Extract a float in [0.0, 1.0] from a PRM model output.

    Searches for the first decimal / integer in the output and clamps it.
    Returns 0.5 (neutral) when no number is found.
    """
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if match:
        value = float(match.group())
        return max(0.0, min(1.0, value))
    return 0.5


def _heuristic_score(step: str) -> float:
    """Lightweight heuristic PRM used when no PRM model is available.

    Scores a step based on the presence of mathematical operations,
    numerical values, and other positive/negative signals.  This is
    intentionally simple – its purpose is to allow the algorithm to run
    end-to-end without loading a second GPU model (e.g. in unit tests or
    CI environments).

    Scoring heuristics
    ------------------
    +0.1 per numeric token (digits with optional decimal point).
    +0.2 if the step contains an assignment-like expression (``=``).
    +0.2 if the step mentions a named operation or function.
    +0.1 if the step contains ``\boxed``.
    −0.3 if the step appears to be just a restatement of the problem.
    Minimum: 0.1, Maximum: 1.0.
    """
    score = 0.3  # baseline
    if re.search(r"\b\d+(?:\.\d+)?\b", step):
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", step)
        score += min(0.2, 0.05 * len(nums))
    if "=" in step:
        score += 0.2
    if re.search(
        r"\b(therefore|thus|so|hence|since|because|calculate|multiply|"
        r"divide|add|subtract|simplify|solve|substitute|factor)\b",
        step,
        re.IGNORECASE,
    ):
        score += 0.15
    if "\\boxed" in step or "the answer is" in step.lower():
        score += 0.1
    # Penalise very short steps.
    if len(step.split()) < 4:
        score -= 0.15
    return max(0.1, min(1.0, score))


def _compute_fitness(step_scores: List[float]) -> float:
    """Aggregate per-step PRM scores into a single fitness value.

    Uses the *harmonic mean* so that a single very low score has a large
    negative impact on the overall fitness, encouraging the algorithm to
    fix weak links rather than masking them with high-scoring neighbours.

    Returns 0.0 for an empty score list.
    """
    if not step_scores:
        return 0.0
    n = len(step_scores)
    # Clamp scores to avoid division by zero.
    clamped = [max(1e-6, s) for s in step_scores]
    harmonic_mean = n / sum(1.0 / s for s in clamped)
    return harmonic_mean


def _is_terminal(steps: List[str]) -> bool:
    """Return ``True`` when the last step contains a boxed final answer."""
    if not steps:
        return False
    last = steps[-1]
    return bool(
        re.search(r"\\boxed\s*\{", last)
        or re.search(r"the\s+answer\s+is\s+\$?[\d\.\-]+", last, re.IGNORECASE)
    )


def _extract_answer(steps: List[str]) -> str:
    """Extract the final boxed answer string from a list of steps."""
    for step in reversed(steps):
        m = re.search(r"\\boxed\s*\{([^}]*)\}", step)
        if m:
            return m.group(1).strip()
        m = re.search(
            r"the\s+answer\s+is\s+\$?([\d\.\-]+)", step, re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
    return ""


# ===========================================================================
# Section 6 – PRSE Algorithm
# ===========================================================================


class PRSEAlgorithm:
    """Process-Reward Guided Surgical Evolution.

    Parameters
    ----------
    model_manager:
        Configured ``LocalModelManager`` instance.
    population_size:
        Number of initial traces to generate (N).
    max_generations:
        Maximum number of evolutionary generations.
    mutation_threshold:
        Step-score threshold τ below which a step is considered "weak"
        and targeted by Surgical Mutation.
    perfect_fitness_threshold:
        When a trace's ``total_fitness`` is at or above this value *and*
        it is terminal, halt the search immediately.
    max_new_tokens:
        Token budget for each individual generation call.
    temperature:
        Sampling temperature for the generator.
    mutation_candidates_per_trace:
        How many mutant offspring to generate per selected parent.
    crossover_candidates:
        How many crossover offspring to generate per parent pair.
    elite_fraction:
        Fraction of the population kept as-is between generations.
    """

    def __init__(
        self,
        model_manager: LocalModelManager,
        population_size: int = 16,
        max_generations: int = 5,
        mutation_threshold: float = 0.6,
        perfect_fitness_threshold: float = 0.95,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        mutation_candidates_per_trace: int = 2,
        crossover_candidates: int = 2,
        elite_fraction: float = 0.25,
    ) -> None:
        self.model_manager = model_manager
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_threshold = mutation_threshold
        self.perfect_fitness_threshold = perfect_fitness_threshold
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.mutation_candidates_per_trace = mutation_candidates_per_trace
        self.crossover_candidates = crossover_candidates
        self.elite_fraction = elite_fraction

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        problem: str,
        tracker: Optional[ComputeTracker] = None,
    ) -> Tuple[ReasoningTrace, ComputeTracker]:
        """Run the full PRSE search for a single problem.

        Parameters
        ----------
        problem:
            The math problem to solve.
        tracker:
            Optional external ``ComputeTracker``; a fresh one is created
            when ``None`` is passed.

        Returns
        -------
        best_trace:
            The highest-fitness ``ReasoningTrace`` found.
        tracker:
            Populated ``ComputeTracker`` for this run.
        """
        if tracker is None:
            tracker = ComputeTracker()

        # Step 1 – Initialise population.
        population = self._initialise(problem, tracker)

        # Step 2 – Evaluate initial population.
        population = self._evaluate_population(problem, population, tracker)

        # Check early stop.
        best = max(population, key=lambda t: t.total_fitness)
        if self._should_stop(best):
            return best, tracker

        # Evolutionary loop.
        for _gen in range(self.max_generations):
            population = self._evolve(problem, population, tracker)
            best = max(population, key=lambda t: t.total_fitness)
            if self._should_stop(best):
                break

        return best, tracker

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(
        self, problem: str, tracker: ComputeTracker
    ) -> List[ReasoningTrace]:
        """Generate initial population of zero-shot CoT traces.

        Parameters
        ----------
        problem:
            The problem to solve.
        tracker:
            Compute tracker to update.

        Returns
        -------
        A list of ``ReasoningTrace`` objects (steps parsed, no PRM scores
        yet).
        """
        prompt = _build_zero_shot_cot_prompt(problem)
        prompts = [prompt] * self.population_size

        t0 = time.perf_counter()
        texts, p_tok, c_tok = self.model_manager.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        latency = time.perf_counter() - t0
        tracker.add_generation(p_tok, c_tok, latency)

        population: List[ReasoningTrace] = []
        for text in texts:
            steps = _parse_steps(text)
            trace = ReasoningTrace(
                problem_statement=problem,
                steps=steps,
                is_terminal=_is_terminal(steps),
            )
            population.append(trace)
        return population

    # ------------------------------------------------------------------
    # Evaluation (PRM scoring)
    # ------------------------------------------------------------------

    def _evaluate_trace(
        self, problem: str, trace: ReasoningTrace, tracker: ComputeTracker
    ) -> ReasoningTrace:
        """Score all steps of a single trace with the PRM."""
        t0 = time.perf_counter()
        scores, prm_tok = self.model_manager.score_steps(problem, trace.steps)
        latency = time.perf_counter() - t0
        tracker.add_prm_call(prm_tok, latency)

        fitness = _compute_fitness(scores)
        return dataclasses.replace(
            trace,
            step_scores=scores,
            total_fitness=fitness,
            is_terminal=_is_terminal(trace.steps),
        )

    def _evaluate_population(
        self,
        problem: str,
        population: List[ReasoningTrace],
        tracker: ComputeTracker,
    ) -> List[ReasoningTrace]:
        """Score every trace in the population in place."""
        return [self._evaluate_trace(problem, t, tracker) for t in population]

    # ------------------------------------------------------------------
    # Evolutionary operators
    # ------------------------------------------------------------------

    def _surgical_mutation(
        self,
        problem: str,
        trace: ReasoningTrace,
        tracker: ComputeTracker,
    ) -> List[ReasoningTrace]:
        """Surgical Mutation operator.

        1. Find step index *k* where score drops below ``mutation_threshold``.
        2. Truncate the trace to steps 0 … k-1 (the validated prefix).
        3. Ask the generator to produce a *different* continuation from
           step k onwards.
        4. Return up to ``mutation_candidates_per_trace`` offspring.

        When all steps are above the threshold (trace is already strong),
        no offspring are produced and an empty list is returned.
        """
        k = trace.first_weak_step(self.mutation_threshold)
        if k is None:
            return []

        # We want at least one valid step before the mutation point.
        prefix_len = max(1, k)
        prefix_steps = trace.prefix(prefix_len)
        prefix_text = trace.prefix_as_text(prefix_len)

        prompt = _build_mutation_prompt(problem, prefix_text, prefix_len + 1)
        prompts = [prompt] * self.mutation_candidates_per_trace

        t0 = time.perf_counter()
        texts, p_tok, c_tok = self.model_manager.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        latency = time.perf_counter() - t0
        tracker.add_generation(p_tok, c_tok, latency)

        # Record mutation telemetry: steps_truncated = total - prefix kept.
        steps_dropped = len(trace.steps) - prefix_len
        for _ in texts:
            tracker.record_mutation(steps_dropped)

        offspring: List[ReasoningTrace] = []
        for text in texts:
            new_steps_raw = _parse_steps(text)
            # Prepend the validated prefix.
            full_steps = prefix_steps + new_steps_raw
            child = ReasoningTrace(
                problem_statement=problem,
                steps=full_steps,
                is_terminal=_is_terminal(full_steps),
            )
            offspring.append(child)
        return offspring

    def _logical_grafting(
        self,
        problem: str,
        parent_a: ReasoningTrace,
        parent_b: ReasoningTrace,
        tracker: ComputeTracker,
    ) -> List[ReasoningTrace]:
        """Logical Grafting (crossover) operator.

        1. Identify Parent A's validated prefix (steps where score ≥ τ).
        2. Identify Parent B's high-scoring later steps as "insights".
        3. Ask the generator to complete Parent A's prefix by integrating
           Parent B's insights.

        If Parent A has no valid prefix or Parent B has no high-scoring
        later steps, an empty list is returned.
        """
        # Build Parent A's valid prefix.
        prefix_a_steps: List[str] = []
        for step, score in zip(parent_a.steps, parent_a.step_scores):
            if score >= self.mutation_threshold:
                prefix_a_steps.append(step)
            else:
                break

        # Build Parent B's high-scoring insights (latter half preferred).
        mid = max(1, len(parent_b.steps) // 2)
        insights_b_steps = [
            step
            for step, score in zip(
                parent_b.steps[mid:], parent_b.step_scores[mid:]
            )
            if score >= self.mutation_threshold
        ]

        if not prefix_a_steps or not insights_b_steps:
            return []

        prefix_a_text = "\n".join(
            f"Step {i + 1}: {s}" for i, s in enumerate(prefix_a_steps)
        )
        insights_b_text = "\n".join(
            f"Insight {i + 1}: {s}" for i, s in enumerate(insights_b_steps)
        )

        prompt = _build_crossover_prompt(
            problem,
            prefix_a_text,
            insights_b_text,
            len(prefix_a_steps) + 1,
        )
        prompts = [prompt] * self.crossover_candidates

        t0 = time.perf_counter()
        texts, p_tok, c_tok = self.model_manager.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        latency = time.perf_counter() - t0
        tracker.add_generation(p_tok, c_tok, latency)

        offspring: List[ReasoningTrace] = []
        for text in texts:
            new_steps_raw = _parse_steps(text)
            full_steps = prefix_a_steps + new_steps_raw
            child = ReasoningTrace(
                problem_statement=problem,
                steps=full_steps,
                is_terminal=_is_terminal(full_steps),
            )
            offspring.append(child)
        tracker.record_crossover()
        return offspring

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------

    def _evolve(
        self,
        problem: str,
        population: List[ReasoningTrace],
        tracker: ComputeTracker,
    ) -> List[ReasoningTrace]:
        """Run one generation of the evolutionary loop.

        Procedure
        ---------
        1. Sort population by fitness (descending).
        2. Keep an elite subset unchanged.
        3. Apply Surgical Mutation to each non-elite trace.
        4. Apply Logical Grafting between the top-2 parents.
        5. Evaluate all offspring.
        6. Merge elites + offspring, truncate to population_size.
        """
        sorted_pop = sorted(population, key=lambda t: t.total_fitness, reverse=True)
        n_elite = max(1, int(self.elite_fraction * self.population_size))
        elites = sorted_pop[:n_elite]
        candidates = sorted_pop[n_elite:]

        offspring: List[ReasoningTrace] = []

        # Surgical Mutation for every non-elite trace.
        for trace in candidates:
            mutants = self._surgical_mutation(problem, trace, tracker)
            offspring.extend(mutants)

        # Logical Grafting between the top two parents.
        if len(sorted_pop) >= 2:
            grafted = self._logical_grafting(
                problem, sorted_pop[0], sorted_pop[1], tracker
            )
            offspring.extend(grafted)

        # Evaluate offspring.
        evaluated_offspring = self._evaluate_population(problem, offspring, tracker)

        # Merge and select.
        combined = elites + evaluated_offspring
        combined.sort(key=lambda t: t.total_fitness, reverse=True)
        return combined[: self.population_size]

    # ------------------------------------------------------------------
    # Stopping criterion
    # ------------------------------------------------------------------

    def _should_stop(self, best: ReasoningTrace) -> bool:
        """Return ``True`` when the best trace is terminal and near-perfect."""
        return (
            best.is_terminal
            and best.total_fitness >= self.perfect_fitness_threshold
        )


# ===========================================================================
# Section 7 – Best-of-N baseline
# ===========================================================================


class BestOfNRunner:
    """Best-of-N baseline: generate N independent traces, return the best.

    Parameters
    ----------
    model_manager:
        Configured ``LocalModelManager`` instance.
    n:
        Number of independent samples to generate.
    max_new_tokens:
        Token budget per sample.
    temperature:
        Sampling temperature.
    """

    def __init__(
        self,
        model_manager: LocalModelManager,
        n: int = 16,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
    ) -> None:
        self.model_manager = model_manager
        self.n = n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def solve(
        self,
        problem: str,
        tracker: Optional[ComputeTracker] = None,
    ) -> Tuple[ReasoningTrace, ComputeTracker]:
        """Generate N traces and return the one with the highest PRM fitness.

        Parameters
        ----------
        problem:
            The math problem to solve.
        tracker:
            Optional external ``ComputeTracker``; a fresh one is created
            when ``None`` is passed.

        Returns
        -------
        best_trace:
            The ``ReasoningTrace`` with the highest ``total_fitness``.
        tracker:
            Populated ``ComputeTracker`` for this run.
        """
        if tracker is None:
            tracker = ComputeTracker()

        prompt = _build_zero_shot_cot_prompt(problem)
        prompts = [prompt] * self.n

        t0 = time.perf_counter()
        texts, p_tok, c_tok = self.model_manager.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        latency = time.perf_counter() - t0
        tracker.add_generation(p_tok, c_tok, latency)

        best: Optional[ReasoningTrace] = None
        for text in texts:
            steps = _parse_steps(text)
            t0 = time.perf_counter()
            scores, prm_tok = self.model_manager.score_steps(problem, steps)
            prm_latency = time.perf_counter() - t0
            tracker.add_prm_call(prm_tok, prm_latency)

            fitness = _compute_fitness(scores)
            trace = ReasoningTrace(
                problem_statement=problem,
                steps=steps,
                step_scores=scores,
                total_fitness=fitness,
                is_terminal=_is_terminal(steps),
            )
            if best is None or trace.total_fitness > best.total_fitness:
                best = trace

        # Should never be None since n >= 1, but satisfy type checker.
        assert best is not None
        return best, tracker



# ===========================================================================
# Section 8 – MATH Dataset loader
# ===========================================================================

#: Difficulty levels to include (Level 4 and Level 5 are the hard tiers).
MATH_TARGET_LEVELS: frozenset = frozenset({"Level 4", "Level 5"})

#: Optional subject filter – ``None`` means all subjects are included.
MATH_TARGET_SUBJECTS: Optional[List[str]] = None


def load_math_problems(
    num_problems: int = 50,
    subjects: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load Level 4 and Level 5 problems from the hendrycks/competition_math dataset.

    The function loads the HuggingFace dataset (cached locally after the first
    download), filters by level and optionally by subject, then returns a
    random subsample of the requested size.

    Parameters
    ----------
    num_problems:
        Maximum number of problems to return.
    subjects:
        Optional whitelist of subjects (e.g. ``["Algebra", "Number Theory"]``).
        When ``None`` all subjects are included.
    seed:
        Random seed for reproducible subsampling.

    Returns
    -------
    A list of dicts with keys:
        ``problem``  – the problem statement (str).
        ``solution`` – the full reference solution (str).
        ``answer``   – the extracted final answer (str).
        ``level``    – difficulty level string, e.g. ``"Level 5"`` (str).
        ``subject``  – problem type, e.g. ``"Algebra"`` (str).

    Raises
    ------
    RuntimeError
        When the ``datasets`` library is not installed.
    """
    if not DATASETS_AVAILABLE:
        raise RuntimeError(
            "The 'datasets' library is required to load MATH problems.  "
            "Run: pip install datasets"
        )

    logger.info(
        "Loading hendrycks/competition_math dataset (levels: %s, subjects: %s, n=%d)",
        sorted(MATH_TARGET_LEVELS),
        subjects or "all",
        num_problems,
    )

    ds = load_dataset("hendrycks/competition_math", split="test")

    records: List[Dict[str, Any]] = []
    for item in ds:
        level: str = item.get("level", "")
        subject: str = item.get("type", item.get("subject", ""))
        if level not in MATH_TARGET_LEVELS:
            continue
        if subjects is not None and subject not in subjects:
            continue

        solution: str = item.get("solution", "")
        # Extract the final \boxed{} answer from the reference solution.
        answer = _extract_answer(_parse_steps(solution)) or solution.strip()

        records.append(
            {
                "problem": item["problem"],
                "solution": solution,
                "answer": answer,
                "level": level,
                "subject": subject,
            }
        )

    logger.info("Found %d problems matching filter criteria.", len(records))

    # Reproducible shuffle + truncate.
    import random
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:num_problems]


# ===========================================================================
# Section 9 – Output / Persistence helpers
# ===========================================================================

# Column order for the per-problem telemetry CSV.
_TELEMETRY_FIELDNAMES: List[str] = [
    "problem_id",
    "level",
    "subject",
    "problem_snippet",
    "ground_truth",
    # Best-of-N columns
    "bon_answer",
    "bon_correct",
    "bon_prompt_tokens",
    "bon_completion_tokens",
    "bon_prm_tokens",
    "bon_total_tokens",
    "bon_wall_clock_s",
    # PRSE columns
    "prse_answer",
    "prse_correct",
    "prse_prompt_tokens",
    "prse_completion_tokens",
    "prse_prm_tokens",
    "prse_total_tokens",
    "prse_wall_clock_s",
    "prse_mutations_applied",
    "prse_crossovers_applied",
    "prse_steps_truncated",
]


def setup_output_dirs(base_dir: Optional[str] = None) -> Dict[str, Path]:
    """Set up output directories, mounting Google Drive if running in Colab.

    When ``base_dir`` is ``None`` the function attempts to detect whether it
    is running inside Google Colab and, if so, mounts Drive and uses
    ``/content/drive/MyDrive/PRSE_MATH_Results/`` as the base.  Outside
    Colab it falls back to ``./prse_math_results/``.

    Parameters
    ----------
    base_dir:
        Override for the root output directory.  Absolute or relative path.

    Returns
    -------
    A dict with keys ``"base"``, ``"plots"`` mapping to ``Path`` objects
    that are guaranteed to exist after this call.
    """
    if base_dir is not None:
        root = Path(base_dir)
    else:
        # Auto-detect Google Colab environment.
        in_colab = "google.colab" in sys.modules or os.path.exists(
            "/content"
        )
        if in_colab:
            try:
                from google.colab import drive  # type: ignore  # noqa: F401
                drive.mount("/content/drive", force_remount=False)
                # Verify the mount succeeded before relying on it.
                drive_ok = os.path.isdir("/content/drive/MyDrive")
                if drive_ok:
                    logger.info("Google Drive mounted at /content/drive")
                else:
                    logger.warning(
                        "Drive mount call succeeded but /content/drive/MyDrive "
                        "is not accessible; falling back to local directory."
                    )
                    root = Path("./prse_math_results")
            except Exception as exc:
                logger.warning("Could not mount Google Drive: %s", exc)
                drive_ok = False
            if drive_ok:
                root = Path("/content/drive/MyDrive/PRSE_MATH_Results")
            else:
                root = Path("./prse_math_results")
        else:
            root = Path("./prse_math_results")

    plots_dir = root / "plots"
    root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", root.resolve())
    return {"base": root, "plots": plots_dir}


def append_telemetry_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append a single per-problem result row to the telemetry CSV.

    Creates the file with a header row on the first call; subsequent calls
    append without repeating the header.  Safe to call after every problem
    so the run can be interrupted without losing prior results.

    Parameters
    ----------
    csv_path:
        Absolute path to the ``.csv`` file.
    row:
        Dict whose keys are a subset of ``_TELEMETRY_FIELDNAMES``.
    """
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=_TELEMETRY_FIELDNAMES, extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_trace_record(jsonl_path: Path, record: Dict[str, Any]) -> None:
    """Append a single trace record to the JSONL traces log.

    Each line is a self-contained JSON object describing the full reasoning
    traces (with step-level PRM scores) for one problem.

    Parameters
    ----------
    jsonl_path:
        Absolute path to the ``.jsonl`` file.
    record:
        Serialisable dict to append.
    """
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ===========================================================================
# Section 10 – Visualisation
# ===========================================================================


def generate_plots(csv_path: Path, plots_dir: Path) -> None:
    """Generate the full academic visualisation suite from the telemetry CSV.

    Requires ``pandas``, ``matplotlib``, and ``seaborn``.  When these are
    unavailable a warning is logged and the function returns immediately.

    The following five plots are produced in both high-resolution PNG and PDF
    formats inside ``plots_dir``:

    1. **accuracy_vs_tokens.{png,pdf}**
       Hero graph: Accuracy vs. total compute (tokens) for BoN and PRSE,
       shown as a scatter/line plot with cumulative token axis.

    2. **accuracy_vs_time.{png,pdf}**
       Accuracy vs. wall-clock time (seconds) comparing both methods.

    3. **compute_breakdown.{png,pdf}**
       Stacked bar chart of generator tokens vs. PRM tokens for PRSE.

    4. **difficulty_degradation.{png,pdf}**
       Grouped bar chart of accuracy by problem level (Level 4 / Level 5)
       for both methods.

    5. **prm_rejection_rate.{png,pdf}**
       Histogram of steps truncated per problem by the Surgical Mutation
       operator.

    Parameters
    ----------
    csv_path:
        Path to ``prse_vs_bon_telemetry.csv`` written by ``main()``.
    plots_dir:
        Directory where plot files will be saved.
    """
    if not PLOTTING_AVAILABLE:
        logger.warning(
            "matplotlib/seaborn not available – skipping plot generation.  "
            "Run: pip install matplotlib seaborn"
        )
        return
    if not PANDAS_AVAILABLE:
        logger.warning(
            "pandas not available – skipping plot generation.  "
            "Run: pip install pandas"
        )
        return
    if not csv_path.exists():
        logger.warning("Telemetry CSV not found at %s – no plots generated.", csv_path)
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("Telemetry CSV is empty – no plots generated.")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    _SAVE_DPI = 300

    def _save(fig: "plt.Figure", name: str) -> None:  # type: ignore[name-defined]
        for ext in ("png", "pdf"):
            dest = plots_dir / f"{name}.{ext}"
            fig.savefig(dest, dpi=_SAVE_DPI, bbox_inches="tight")
            logger.info("Saved plot: %s", dest)
        plt.close(fig)

    # ---------------------------------------------------------------
    # 1. Accuracy vs. Total Compute (Tokens) – HERO GRAPH
    # ---------------------------------------------------------------
    df_sorted = df.sort_values("bon_total_tokens")
    bon_cum_tokens = df_sorted["bon_total_tokens"].cumsum()
    prse_cum_tokens = df_sorted["prse_total_tokens"].cumsum()
    bon_cum_acc = df_sorted["bon_correct"].expanding().mean() * 100
    prse_cum_acc = df_sorted["prse_correct"].expanding().mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bon_cum_tokens, bon_cum_acc, marker="o", label="Best-of-N", color="#2196F3")
    ax.plot(prse_cum_tokens, prse_cum_acc, marker="s", label="PRSE", color="#FF5722")
    ax.set_xlabel("Cumulative Tokens Consumed")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Total Compute (Tokens)")
    ax.legend()
    _save(fig, "accuracy_vs_tokens")

    # ---------------------------------------------------------------
    # 2. Accuracy vs. Wall-Clock Time
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    bon_cum_time = df_sorted["bon_wall_clock_s"].cumsum()
    prse_cum_time = df_sorted["prse_wall_clock_s"].cumsum()
    ax.plot(bon_cum_time, bon_cum_acc, marker="o", label="Best-of-N", color="#2196F3")
    ax.plot(prse_cum_time, prse_cum_acc, marker="s", label="PRSE", color="#FF5722")
    ax.set_xlabel("Cumulative Wall-Clock Time (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Wall-Clock Time")
    ax.legend()
    _save(fig, "accuracy_vs_time")

    # ---------------------------------------------------------------
    # 3. Compute Breakdown (PRSE): Generator Tokens vs. PRM Tokens
    # ---------------------------------------------------------------
    prse_gen = df["prse_prompt_tokens"] + df["prse_completion_tokens"]
    prse_prm = df["prse_prm_tokens"]
    problem_ids = range(len(df))

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.4), 5))
    ax.bar(problem_ids, prse_gen, label="Generator Tokens", color="#4CAF50")
    ax.bar(problem_ids, prse_prm, bottom=prse_gen, label="PRM Tokens", color="#FF9800")
    ax.set_xlabel("Problem Index")
    ax.set_ylabel("Token Count")
    ax.set_title("PRSE Compute Breakdown per Problem")
    ax.legend()
    _save(fig, "compute_breakdown")

    # ---------------------------------------------------------------
    # 4. Difficulty Degradation: Accuracy by Level for BoN vs. PRSE
    # ---------------------------------------------------------------
    if "level" in df.columns:
        level_stats = (
            df.groupby("level")[["bon_correct", "prse_correct"]]
            .mean()
            .mul(100)
            .reset_index()
        )
        x = range(len(level_stats))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(
            [i - width / 2 for i in x],
            level_stats["bon_correct"],
            width,
            label="Best-of-N",
            color="#2196F3",
        )
        ax.bar(
            [i + width / 2 for i in x],
            level_stats["prse_correct"],
            width,
            label="PRSE",
            color="#FF5722",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(level_stats["level"].tolist())
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by Difficulty Level")
        ax.legend()
        _save(fig, "difficulty_degradation")

    # ---------------------------------------------------------------
    # 5. PRM Rejection Rate: steps truncated per problem
    # ---------------------------------------------------------------
    if "prse_steps_truncated" in df.columns:
        col = df["prse_steps_truncated"].dropna()
        max_val = int(col.max()) if not col.empty and col.max() == col.max() else 0
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(
            col,
            bins=max(5, max_val + 1),
            color="#9C27B0",
            edgecolor="white",
        )
        ax.set_xlabel("Steps Truncated by Surgical Mutation")
        ax.set_ylabel("Number of Problems")
        ax.set_title("PRM Rejection Rate Distribution")
        _save(fig, "prm_rejection_rate")

    logger.info("All plots saved to %s", plots_dir)


# ===========================================================================
# Section 11 – Answer checker (shared utility)
# ===========================================================================


def _check_answer(predicted: str, ground_truth: str) -> bool:
    """Lightweight answer checker: compare numeric values when possible.

    Falls back to exact string match when numeric conversion fails.
    """
    pred = predicted.strip().lstrip("$").rstrip(".")
    gt = ground_truth.strip().lstrip("$").rstrip(".")
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return pred.lower() == gt.lower()


# ===========================================================================
# Section 12 – Main execution block
# ===========================================================================


def main() -> None:
    """Run PRSE vs Best-of-N on Level 4/5 MATH problems and save results.

    Pipeline
    --------
    1. Mount Google Drive (when running in Colab) and set up output dirs.
    2. Load Level 4 and Level 5 problems from the ``hendrycks/competition_math`` dataset.
    3. Initialise vLLM models (generator + PRM) on the H100.
    4. For each problem:
       a. Run Best-of-N and record telemetry.
       b. Run PRSE and record telemetry.
       c. **Immediately** append one row to ``prse_vs_bon_telemetry.csv``
          and one record to ``raw_evolutionary_traces.jsonl``.
    5. On completion (or ``KeyboardInterrupt``): write ``summary_metrics.json``
       and call ``generate_plots()`` to produce all visualisation artefacts.

    Output files (all under the configured base directory)
    ------------------------------------------------------
    ``prse_vs_bon_telemetry.csv``   – per-problem breakdown.
    ``raw_evolutionary_traces.jsonl`` – full reasoning traces with PRM scores.
    ``summary_metrics.json``        – aggregate statistics.
    ``plots/``                      – all PNG/PDF graphs.
    """
    logger.info("=" * 70)
    logger.info("Process-Reward Guided Surgical Evolution (PRSE) – MATH Benchmark")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Set up output directories (auto-mount Drive when in Colab).
    # ------------------------------------------------------------------
    dirs = setup_output_dirs()
    base_dir: Path = dirs["base"]
    plots_dir: Path = dirs["plots"]
    csv_path = base_dir / "prse_vs_bon_telemetry.csv"
    jsonl_path = base_dir / "raw_evolutionary_traces.jsonl"
    summary_path = base_dir / "summary_metrics.json"

    # ------------------------------------------------------------------
    # 2. Load dataset.
    # ------------------------------------------------------------------
    problems = load_math_problems(num_problems=50)
    logger.info("Loaded %d MATH problems (Level 4 & 5).", len(problems))

    # ------------------------------------------------------------------
    # 3. Initialise models (H100-optimised: 55 % generator, 35 % PRM).
    # ------------------------------------------------------------------
    manager = LocalModelManager(
        generator_model_id="Qwen/Qwen2.5-Math-7B-Instruct",
        prm_model_id="peiyi9979/math-shepherd-mistral-7b-prm",
        tensor_parallel_size=1,
        max_model_len=8192,
        generator_gpu_memory_utilisation=0.55,
        prm_gpu_memory_utilisation=0.35,
    )

    bon_runner = BestOfNRunner(
        model_manager=manager,
        n=16,
        max_new_tokens=2048,
        temperature=0.8,
    )
    prse_runner = PRSEAlgorithm(
        model_manager=manager,
        population_size=16,
        max_generations=5,
        mutation_threshold=0.6,
        perfect_fitness_threshold=0.95,
        max_new_tokens=2048,
        temperature=0.8,
    )

    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 4. Main problem loop.
    # ------------------------------------------------------------------
    try:
        for prob_idx, problem_dict in enumerate(problems):
            problem: str = problem_dict["problem"]
            ground_truth: str = problem_dict["answer"]
            level: str = problem_dict["level"]
            subject: str = problem_dict["subject"]

            logger.info(
                "[%d/%d] %s | %s | %s…",
                prob_idx + 1,
                len(problems),
                level,
                subject,
                problem[:80],
            )

            # --- Best-of-N ---
            t_bon = time.perf_counter()
            bon_trace, bon_tracker = bon_runner.solve(problem)
            bon_wall = time.perf_counter() - t_bon
            bon_answer = _extract_answer(bon_trace.steps)
            bon_correct = _check_answer(bon_answer, ground_truth)
            bon_summary = bon_tracker.summary()
            logger.info(
                "  BoN  → answer=%r  correct=%s  tokens=%d  t=%.1fs",
                bon_answer,
                bon_correct,
                bon_summary["total_tokens"],
                bon_wall,
            )

            # --- PRSE ---
            t_prse = time.perf_counter()
            prse_trace, prse_tracker = prse_runner.solve(problem)
            prse_wall = time.perf_counter() - t_prse
            prse_answer = _extract_answer(prse_trace.steps)
            prse_correct = _check_answer(prse_answer, ground_truth)
            prse_summary = prse_tracker.summary()
            logger.info(
                "  PRSE → answer=%r  correct=%s  tokens=%d  t=%.1fs"
                "  mutations=%d  crossovers=%d  steps_truncated=%d",
                prse_answer,
                prse_correct,
                prse_summary["total_tokens"],
                prse_wall,
                prse_summary["mutations_applied"],
                prse_summary["crossovers_applied"],
                prse_summary["steps_truncated"],
            )

            # --- Build telemetry row ---
            row: Dict[str, Any] = {
                "problem_id": prob_idx,
                "level": level,
                "subject": subject,
                "problem_snippet": problem[:120],
                "ground_truth": ground_truth,
                # BoN
                "bon_answer": bon_answer,
                "bon_correct": bon_correct,
                "bon_prompt_tokens": bon_summary["prompt_tokens"],
                "bon_completion_tokens": bon_summary["completion_tokens"],
                "bon_prm_tokens": bon_summary["prm_tokens"],
                "bon_total_tokens": bon_summary["total_tokens"],
                "bon_wall_clock_s": round(bon_wall, 3),
                # PRSE
                "prse_answer": prse_answer,
                "prse_correct": prse_correct,
                "prse_prompt_tokens": prse_summary["prompt_tokens"],
                "prse_completion_tokens": prse_summary["completion_tokens"],
                "prse_prm_tokens": prse_summary["prm_tokens"],
                "prse_total_tokens": prse_summary["total_tokens"],
                "prse_wall_clock_s": round(prse_wall, 3),
                "prse_mutations_applied": prse_summary["mutations_applied"],
                "prse_crossovers_applied": prse_summary["crossovers_applied"],
                "prse_steps_truncated": prse_summary["steps_truncated"],
            }
            rows.append(row)

            # --- Incremental saves (survive Colab timeouts) ---
            append_telemetry_row(csv_path, row)

            trace_record: Dict[str, Any] = {
                "problem_id": prob_idx,
                "problem": problem,
                "ground_truth": ground_truth,
                "level": level,
                "subject": subject,
                "bon": {
                    "answer": bon_answer,
                    "correct": bon_correct,
                    "steps": bon_trace.steps,
                    "step_scores": bon_trace.step_scores,
                    "total_fitness": bon_trace.total_fitness,
                },
                "prse": {
                    "answer": prse_answer,
                    "correct": prse_correct,
                    "steps": prse_trace.steps,
                    "step_scores": prse_trace.step_scores,
                    "total_fitness": prse_trace.total_fitness,
                },
            }
            append_trace_record(jsonl_path, trace_record)

    except KeyboardInterrupt:
        logger.warning("Run interrupted by user.  Saving summary and plots…")

    # ------------------------------------------------------------------
    # 5. Save aggregate summary and generate plots.
    # ------------------------------------------------------------------
    if rows:
        n = len(rows)
        bon_acc = sum(r["bon_correct"] for r in rows) / n
        prse_acc = sum(r["prse_correct"] for r in rows) / n
        summary: Dict[str, Any] = {
            "n_problems": n,
            "bon_accuracy": round(bon_acc, 4),
            "prse_accuracy": round(prse_acc, 4),
            "bon_total_tokens": sum(r["bon_total_tokens"] for r in rows),
            "prse_total_tokens": sum(r["prse_total_tokens"] for r in rows),
            "bon_total_wall_clock_s": round(
                sum(r["bon_wall_clock_s"] for r in rows), 2
            ),
            "prse_total_wall_clock_s": round(
                sum(r["prse_wall_clock_s"] for r in rows), 2
            ),
        }
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Summary saved to %s", summary_path)
        logger.info(
            "Final results — BoN accuracy: %.1f%%  PRSE accuracy: %.1f%%",
            bon_acc * 100,
            prse_acc * 100,
        )

    generate_plots(csv_path, plots_dir)


if __name__ == "__main__":
    main()
