#!/usr/bin/env python3
"""
Collect Pythia evaluation data from GitHub and save as .npy files.

This script downloads evaluation JSONs from the EleutherAI/pythia GitHub repo
and converts them to the same .npy format used by DataDecide evaluations.

Output format: 2D numpy array with shape (2, num_steps)
  - Row 0: training step numbers
  - Row 1: accuracy values

Usage:
    python scripts/collect_pythia_evals.py [--models MODEL1 MODEL2 ...] [--output-dir DIR]
"""

import argparse
import json
import numpy as np
import requests
from pathlib import Path
from typing import Optional


# Mapping from Pythia benchmark names to DataDecide task names
# hendrycksTest-* in Pythia corresponds to mmlu_* in DataDecide
BENCHMARK_MAPPING = {
    # Direct mappings
    "arc_challenge": "arc_challenge",
    "arc_easy": "arc_easy",
    "piqa": "piqa",
    "winogrande": "winogrande",
    # MMLU mappings (hendrycksTest-* -> mmlu_*)
    "hendrycksTest-abstract_algebra": "mmlu_abstract_algebra",
    "hendrycksTest-anatomy": "mmlu_anatomy",
    "hendrycksTest-astronomy": "mmlu_astronomy",
    "hendrycksTest-business_ethics": "mmlu_business_ethics",
    "hendrycksTest-clinical_knowledge": "mmlu_clinical_knowledge",
    "hendrycksTest-college_biology": "mmlu_college_biology",
    "hendrycksTest-college_chemistry": "mmlu_college_chemistry",
    "hendrycksTest-college_computer_science": "mmlu_college_computer_science",
    "hendrycksTest-college_mathematics": "mmlu_college_mathematics",
    "hendrycksTest-college_medicine": "mmlu_college_medicine",
    "hendrycksTest-college_physics": "mmlu_college_physics",
    "hendrycksTest-computer_security": "mmlu_computer_security",
    "hendrycksTest-conceptual_physics": "mmlu_conceptual_physics",
    "hendrycksTest-econometrics": "mmlu_econometrics",
    "hendrycksTest-electrical_engineering": "mmlu_electrical_engineering",
    "hendrycksTest-elementary_mathematics": "mmlu_elementary_mathematics",
    "hendrycksTest-formal_logic": "mmlu_formal_logic",
    "hendrycksTest-global_facts": "mmlu_global_facts",
    "hendrycksTest-high_school_biology": "mmlu_high_school_biology",
    "hendrycksTest-high_school_chemistry": "mmlu_high_school_chemistry",
    "hendrycksTest-high_school_computer_science": "mmlu_high_school_computer_science",
    "hendrycksTest-high_school_european_history": "mmlu_high_school_european_history",
    "hendrycksTest-high_school_geography": "mmlu_high_school_geography",
    "hendrycksTest-high_school_government_and_politics": "mmlu_high_school_government_and_politics",
    "hendrycksTest-high_school_macroeconomics": "mmlu_high_school_macroeconomics",
    "hendrycksTest-high_school_mathematics": "mmlu_high_school_mathematics",
    "hendrycksTest-high_school_microeconomics": "mmlu_high_school_microeconomics",
    "hendrycksTest-high_school_physics": "mmlu_high_school_physics",
    "hendrycksTest-high_school_psychology": "mmlu_high_school_psychology",
    "hendrycksTest-high_school_statistics": "mmlu_high_school_statistics",
    "hendrycksTest-high_school_us_history": "mmlu_high_school_us_history",
    "hendrycksTest-high_school_world_history": "mmlu_high_school_world_history",
    "hendrycksTest-human_aging": "mmlu_human_aging",
    "hendrycksTest-human_sexuality": "mmlu_human_sexuality",
    "hendrycksTest-international_law": "mmlu_international_law",
    "hendrycksTest-jurisprudence": "mmlu_jurisprudence",
    "hendrycksTest-logical_fallacies": "mmlu_logical_fallacies",
    "hendrycksTest-machine_learning": "mmlu_machine_learning",
    "hendrycksTest-management": "mmlu_management",
    "hendrycksTest-marketing": "mmlu_marketing",
    "hendrycksTest-medical_genetics": "mmlu_medical_genetics",
    "hendrycksTest-miscellaneous": "mmlu_miscellaneous",
    "hendrycksTest-moral_disputes": "mmlu_moral_disputes",
    "hendrycksTest-moral_scenarios": "mmlu_moral_scenarios",
    "hendrycksTest-nutrition": "mmlu_nutrition",
    "hendrycksTest-philosophy": "mmlu_philosophy",
    "hendrycksTest-prehistory": "mmlu_prehistory",
    "hendrycksTest-professional_accounting": "mmlu_professional_accounting",
    "hendrycksTest-professional_law": "mmlu_professional_law",
    "hendrycksTest-professional_medicine": "mmlu_professional_medicine",
    "hendrycksTest-professional_psychology": "mmlu_professional_psychology",
    "hendrycksTest-public_relations": "mmlu_public_relations",
    "hendrycksTest-security_studies": "mmlu_security_studies",
    "hendrycksTest-sociology": "mmlu_sociology",
    "hendrycksTest-us_foreign_policy": "mmlu_us_foreign_policy",
    "hendrycksTest-virology": "mmlu_virology",
    "hendrycksTest-world_religions": "mmlu_world_religions",
}

# DataDecide tasks that are NOT available in Pythia evals
# (hellaswag, boolq, csqa, socialiqa, openbookqa are missing from Pythia zero-shot evals)
DATADECIDE_ONLY_TASKS = {"hellaswag", "boolq", "csqa", "socialiqa", "openbookqa"}

# Available Pythia model sizes
PYTHIA_MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com/repos/EleutherAI/pythia/contents/evals/pythia-v1"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/EleutherAI/pythia/main/evals/pythia-v1"


def get_available_steps(model: str) -> list[int]:
    """Get list of available checkpoint steps for a model from GitHub."""
    url = f"{GITHUB_API_BASE}/{model}/zero-shot"
    response = requests.get(url)
    response.raise_for_status()

    files = response.json()
    steps = []
    for f in files:
        name = f["name"]
        if name.endswith(".json"):
            # Parse step from filename like "1.4b_step33000.json"
            step_str = name.split("_step")[1].replace(".json", "")
            steps.append(int(step_str))

    return sorted(steps)


def download_eval_json(model: str, step: int) -> dict:
    """Download evaluation JSON for a specific model and step."""
    # Get model size string for filename (e.g., "1.4b" from "pythia-1.4b")
    size_str = model.replace("pythia-", "")
    filename = f"{size_str}_step{step}.json"
    url = f"{GITHUB_RAW_BASE}/{model}/zero-shot/{filename}"

    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def extract_accuracies(eval_data: dict) -> dict[str, float]:
    """Extract accuracy values from evaluation JSON, mapped to DataDecide task names."""
    results = eval_data.get("results", {})
    accuracies = {}

    for benchmark, metrics in results.items():
        if benchmark in BENCHMARK_MAPPING:
            task_name = BENCHMARK_MAPPING[benchmark]
            # Use acc_norm if available (for MMLU), otherwise use acc
            if "acc_norm" in metrics:
                accuracies[task_name] = metrics["acc_norm"]
            elif "acc" in metrics:
                accuracies[task_name] = metrics["acc"]

    return accuracies


def collect_model_evals(model: str, output_dir: Path, verbose: bool = True) -> dict[str, np.ndarray]:
    """Collect all evaluations for a model and save as .npy files."""
    if verbose:
        print(f"\nCollecting evaluations for {model}...")

    # Get available steps
    try:
        steps = get_available_steps(model)
    except requests.exceptions.HTTPError as e:
        print(f"  Error getting steps for {model}: {e}")
        return {}

    if verbose:
        print(f"  Found {len(steps)} checkpoints: {steps[:5]}...{steps[-3:]}")

    # Collect accuracies for each step
    all_accuracies: dict[str, list[tuple[int, float]]] = {}

    for step in steps:
        try:
            eval_data = download_eval_json(model, step)
            accuracies = extract_accuracies(eval_data)

            for task, acc in accuracies.items():
                if task not in all_accuracies:
                    all_accuracies[task] = []
                all_accuracies[task].append((step, acc))

            if verbose:
                print(f"  Step {step}: {len(accuracies)} tasks")
        except requests.exceptions.HTTPError as e:
            print(f"  Error downloading step {step}: {e}")
            continue

    # Convert to numpy arrays and save
    model_dir = output_dir / model.replace("pythia-", "EleutherAI--pythia-")
    model_dir.mkdir(parents=True, exist_ok=True)

    saved_tasks = {}
    for task, step_acc_pairs in all_accuracies.items():
        # Sort by step
        step_acc_pairs.sort(key=lambda x: x[0])
        steps_arr = np.array([s for s, _ in step_acc_pairs], dtype=np.float64)
        accs_arr = np.array([a for _, a in step_acc_pairs], dtype=np.float64)

        # Create 2D array matching DataDecide format
        arr = np.stack([steps_arr, accs_arr])

        # Save to file
        output_path = model_dir / f"{task}.npy"
        np.save(output_path, arr)
        saved_tasks[task] = arr

    if verbose:
        print(f"  Saved {len(saved_tasks)} task files to {model_dir}")

    return saved_tasks


def main():
    parser = argparse.ArgumentParser(description="Collect Pythia evaluation data from GitHub")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to collect (default: all). Available: {', '.join(PYTHIA_MODELS)}"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hf_eval"),
        help="Output directory for .npy files (default: results/hf_eval)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    models = args.models or PYTHIA_MODELS
    output_dir = args.output_dir
    verbose = not args.quiet

    # Validate models
    for model in models:
        if model not in PYTHIA_MODELS:
            print(f"Warning: {model} not in known models list")

    if verbose:
        print(f"Collecting evaluations for {len(models)} models...")
        print(f"Output directory: {output_dir}")
        print(f"\nNote: The following DataDecide tasks are NOT available in Pythia:")
        print(f"  {', '.join(sorted(DATADECIDE_ONLY_TASKS))}")

    # Collect evaluations for each model
    all_results = {}
    for model in models:
        results = collect_model_evals(model, output_dir, verbose=verbose)
        all_results[model] = results

    if verbose:
        print("\n" + "="*60)
        print("Summary:")
        for model, tasks in all_results.items():
            print(f"  {model}: {len(tasks)} tasks")

        # Show which tasks are available
        all_tasks = set()
        for tasks in all_results.values():
            all_tasks.update(tasks.keys())
        print(f"\nTotal unique tasks collected: {len(all_tasks)}")
        print(f"Tasks: {', '.join(sorted(all_tasks)[:10])}...")


if __name__ == "__main__":
    main()
