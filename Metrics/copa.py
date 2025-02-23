import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


def copa_metric(predictions, references):
    """
    Computes the COPA metric (accuracy) for the given predictions.

    This function uses the `evaluate` library to calculate the accuracy of the predictions based on the COPA task from SuperGLUE.

    Args:
        predictions (list): A list of predictions as integers.
        references (list): A list of reference labels as integers.

    Returns:
        dict: A dictionary containing the computed metrics (e.g., accuracy).
    """
    metric = load("super_glue", "copa")
    results = metric.compute(predictions=predictions, references=references)
    return results


def compute_copa(model, subset_length=10):
    """
    Evaluates a model on a subset of the COPA dataset from SuperGLUE and saves the results to a file.

    This function loads a subset of the COPA dataset, uses the model to generate predictions, saves the predictions
    and reference answers to a JSON file, and then computes the COPA metric.

    Args:
        model (function): A function that calls the model and returns an answer for the given text.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the computed COPA metrics.
    """
    # Load the COPA dataset.
    dataset = dl.load_dataset("copa")
    subset = dataset[:subset_length]

    predictions = []
    references = []

    # Iterate over the subset of the dataset.
    for entry in subset:
        premise = entry['premise']
        choice1 = entry['choice1']
        choice2 = entry['choice2']
        question = entry['question']  # Either "cause" or "effect"
        label = entry['label']  # 0 or 1 indicating the correct choice

        # Create the input text for the model.
        input_text = (
            f"Premise: {premise}\n"
            f"Question: What is the {'cause' if question == 'cause' else 'effect'}?\n"
            f"Choice 1: {choice1}\n"
            f"Choice 2: {choice2}\n"
            f"Answer with either 0 for Choice 1 or 1 for Choice 2. Do not answer with text! An example answer would be: \"1\""
        )

        try:
            # Call the model with the input text.
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

    # Save the predictions and references to a JSON file.
    output_data = {
        "predictions": predictions,
        "references": references
    }
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "copa_predictions_references.json")
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the COPA metrics.
    results = copa_metric(predictions, references)
    return results


def copa_eval(models, subset_length=10):
    """
    Evaluates COPA performance for all provided models and saves the aggregated results in a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the evaluation results for all models.
    """
    all_results = {}
    for model in models:
        result = compute_copa(model, subset_length)
        all_results[model.model_name] = result

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "copa_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)
    print(f"All COPA results have been saved to {results_file}.")

    return all_results
