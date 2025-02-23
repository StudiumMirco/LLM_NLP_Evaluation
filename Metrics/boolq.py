import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


def boolq_metric(predictions, references):
    """
    Computes the BoolQ metric (accuracy) for the given predictions.

    This function uses the `evaluate` library to compute the accuracy based on the BoolQ task from SuperGLUE.

    Args:
        predictions (list): A list of predicted answers as integers.
        references (list): A list of reference answers as integers.

    Returns:
        dict: A dictionary containing the computed metrics (e.g., accuracy).
    """
    metric = load("super_glue", "boolq")
    results = metric.compute(predictions=predictions, references=references)
    return results


def compute_boolq(model, subset_length=10):
    """
    Evaluates a model on a subset of the BoolQ dataset from SuperGLUE and saves the predictions and reference answers.

    This function loads a subset of the BoolQ dataset, uses the model to generate answers, saves both the predictions
    and the reference answers in a JSON file, and computes the BoolQ metric.

    Args:
        model (function): A function that calls the model and returns an answer for the given text.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the computed BoolQ metrics.
    """
    # Load the BoolQ dataset.
    dataset = dl.load_dataset("boolq")
    subset = dataset[:subset_length]

    predictions = []
    references = []

    # Iterate over the dataset subset.
    for entry in subset:
        question = entry['question']
        passage = entry['passage']
        label = entry['label']  # 0 = "false", 1 = "true"

        # Create the input prompt for the model.
        input_text = (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            "Answer with either 0 for 'false' or 1 for 'true'. Do not answer with text!"
        )

        try:
            # Call the model with the input prompt.
            response = model(input_text, max_new_tokens=50)
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        try:
            # Convert the model's response to an integer.
            prediction_label = int(response.strip())
        except Exception as e:
            print(model.model_name)
            print(response)
            print("Incorrect response format - skipping entry")
            print(e)
            continue

        predictions.append(prediction_label)
        references.append(label)

    # Save predictions and references to a JSON file.
    output_data = {
        "predictions": predictions,
        "references": references
    }
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "boolq_predictions_references.json")
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the BoolQ metric.
    results = boolq_metric(predictions, references)
    return results


def boolq_eval(models, subset_length=10):
    """
    Evaluates the BoolQ task for all provided models and saves the aggregated results in a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for each model.

    Returns:
        dict: A dictionary containing the evaluation results for all models.
    """
    all_results = {}
    for model in models:
        result = compute_boolq(model, subset_length)
        all_results[model.model_name] = result

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "boolq_all_models_results.json")
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)
    print(f"All BoolQ results have been saved to {results_file}.")

    return all_results
