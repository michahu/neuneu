"""Analysis tools for the meta-model."""

from .utils import (
    logistic_function,
    fit_logistic_scaling_law,
    predict_logistic,
    compute_metrics,
    compute_errors,
    compute_errors_per_timestep,
)

# Classic LM eval tasks
CLASSIC_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "csqa",
    "piqa",
    "winogrande",
    "openbookqa"
]

# Chance accuracies for downstream tasks (1/num_choices)
TASK_CHANCE_ACCURACY = {
    "hellaswag": 0.25,      # 4-choice
    "arc_easy": 0.25,       # 4-choice
    "arc_challenge": 0.25,  # 4-choice
    "openbookqa": 0.25,     # 4-choice
    "csqa": 0.20,           # 5-choice
    "boolq": 0.50,          # 2-choice
    "piqa": 0.50,           # 2-choice
    "winogrande": 0.50,     # 2-choice
}
DEFAULT_CHANCE_ACCURACY = 0.25  # Fallback for unknown tasks

__all__ = [
    # Utils
    "logistic_function",
    "fit_logistic_scaling_law",
    "predict_logistic",
    "compute_metrics",
    "compute_errors",
    "compute_errors_per_timestep",
    # Constants
    "CLASSIC_TASKS",
    "TASK_CHANCE_ACCURACY",
    "DEFAULT_CHANCE_ACCURACY",
]
