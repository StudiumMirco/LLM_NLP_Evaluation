import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


def axb_metric(predictions, references):
    """
    Computes the AX-B metric (accuracy) for the given predictions.

    This function uses the `evaluate` library to calculate the metric for the AX-B task from SuperGLUE.

    Args:
        predictions (list): A list of predictions as integers.
        references (list): A list of reference labels as integers.

    Returns:
        dict: A dictionary containing the computed metric results.
    """
    metric = load("super_glue", "axb")
    results = metric.compute(predictions=predictions, references=references)
    return results


def compute_axb(model, subset_length=10):
    """
    Evaluates a model on a subset of the AX-B dataset from SuperGLUE and saves the results to a file.

    This function loads a subset of the AX-B dataset, uses the model to generate predictions, saves the
    predictions and reference answers in a JSON file, and computes the AX-B metric.

    Args:
        model (function): A function that calls the model and returns an answer for the given text.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Load the AX-B dataset
    dataset = dl.load_dataset("axb")  # Instantiate the AX-B dataset loader
    subset = dataset[:subset_length]

    predictions = []
    references = []

    # Iterate over the subset of the dataset
    for entry in subset:
        sentence1 = entry['sentence1']
        sentence2 = entry['sentence2']
        label = entry['label']  # 0 = "not entailment", 1 = "entailment"

        # Create the input text for the model
        input_text = (
            f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\n"
            "Does sentence 1 entail sentence 2? Answer with either 1 for 'not entailment' or 0 for 'entailment'. Do not answer with text!"
        )
        try:
            response = model(input_text, max_new_tokens=50)
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        try:
            prediction_label = int(response.strip())
        except Exception as e:
            print(model.model_name)
            print(response)
            print("Incorrect response format - skipping entry")
            print(e)
            continue

        predictions.append(prediction_label)
        references.append(label)

    # Save the predictions and references to a JSON file
    output_data = {
        "predictions": predictions,
        "references": references
    }
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "axb_predictions_references.json")

    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the AX-B metric
    results = axb_metric(predictions, references)
    return results


def axb_eval(models, subset_length=10):
    """
    Evaluates the AX-B task for all provided models and saves the aggregated results to a JSON file.
    AX-B is an extension of the core SuperGLUE metrics.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for each model's evaluation.

    Returns:
        dict: A dictionary containing the evaluation results for all models.
    """
    all_results = {}
    for model in models:
        result = compute_axb(model, subset_length)
        all_results[model.model_name] = result

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "axb_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)
    print(f"All AX-B results have been saved to {results_file}.")

    return all_results
