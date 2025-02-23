from Metrics.axb import axb_eval
from Metrics.axg import axg_eval
from Metrics.boolq import boolq_eval
from Metrics.cb import cb_eval
from Metrics.copa import copa_eval
from Metrics.multirc import multirc_eval
from Metrics.record import record_eval
from Metrics.rte import rte_eval
from Metrics.wic import wic_eval
from Metrics.wsc import wsc_eval
import os
import json


def superGLUE(models):
    """
    Runs evaluations on various SuperGLUE tasks using the provided models,
    aggregates the task-specific results, computes overall scores, and saves
    the aggregated results to a JSON file.

    Evaluated tasks:
      - axb, axg, cb, copa, rte: entire datasets are used.
      - boolq, multirc, record: only the first 1000 entries are used.
      - wic, wsc: entire datasets are used.

    Args:
        models (list): A list of model instances to be evaluated.
    """
    # Obtain results from each SuperGLUE task evaluation.
    results = {
        "axb": axb_eval(models, subset_length=-1),  # Entire dataset
        "axg": axg_eval(models, subset_length=-1),  # Entire dataset
        "boolq": boolq_eval(models, subset_length=1000),  # First 1000 entries
        "cb": cb_eval(models, subset_length=-1),  # Entire dataset
        "copa": copa_eval(models, subset_length=-1),  # Entire dataset
        "multirc": multirc_eval(models, subset_length=1000),  # First 1000 entries
        "record": record_eval(models, subset_length=1000),  # First 1000 entries
        "rte": rte_eval(models, subset_length=-1),  # Entire dataset
        "wic": wic_eval(models, subset_length=-1),  # Entire dataset
        "wsc": wsc_eval(models, subset_length=-1)  # Entire dataset
    }

    print(results)

    # Calculate overall SuperGLUE scores based on task-specific metrics.
    superglue_results = calculate_superglue_scores(results)

    # Remove 'num_tasks' key since it is only used for calculating averages.
    for model in superglue_results:
        del superglue_results[model]["num_tasks"]

    # Save the aggregated results to a JSON file.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "superGLUE_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump(superglue_results, file, indent=4)

    print(f"SuperGLUE results have been saved to {results_file}")


def calculate_superglue_scores(results):
    """
    Calculates overall SuperGLUE scores for each model by aggregating task-specific scores.

    For each model, the task score is computed as the average of all metric values (scaled to 0-100).
    The overall score is then computed as the average of the task scores.

    Args:
        results (dict): A dictionary where each key is a task name and each value is a dict
                        of model evaluation metrics for that task.

    Returns:
        dict: A dictionary with overall SuperGLUE scores for each model.
    """
    superglue_results = {}

    # Iterate through each task and accumulate scores for each model.
    for task, models in results.items():
        for model, metrics in models.items():
            # Calculate the task score as the average of all metric values, scaled to 0-100.
            task_score = (sum(metrics.values()) / len(metrics)) * 100

            # Initialize model entry if it doesn't exist.
            if model not in superglue_results:
                superglue_results[model] = {"task_scores": {}, "overall_score": 0, "num_tasks": 0}

            # Record the task score and update overall score and task count.
            superglue_results[model]["task_scores"][task] = task_score
            superglue_results[model]["overall_score"] += task_score
            superglue_results[model]["num_tasks"] += 1

    # Compute overall score per model as the average over all tasks.
    for model in superglue_results:
        superglue_results[model]["overall_score"] = (
                superglue_results[model]["overall_score"] / superglue_results[model]["num_tasks"]
        )

    return superglue_results
