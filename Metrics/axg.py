import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


def axg_metric(predictions, references):
    """
    Computes the AX-G metric (accuracy) for the given predictions.

    This function loads the AX-G metric from the evaluate library (part of the SuperGLUE extension)
    and computes the metric using the provided predictions and reference labels.

    Args:
        predictions (list): A list of predictions as integers.
        references (list): A list of reference labels as integers.

    Returns:
        dict: A dictionary containing the computed metric results.
    """
    metric = load("super_glue", "axg")
    results = metric.compute(predictions=predictions, references=references)
    return results


def compute_axg(model, subset_length=10):
    """
    Evaluates a model on a subset of the AX-G dataset from SuperGLUE and saves the results to a file.

    This function loads a subset of the AX-G dataset, uses the provided model to generate predictions,
    and saves both the predictions and the reference labels in a JSON file. Finally, it computes the AX-G metric.

    Args:
        model (function): A function that calls the model and returns an answer for the given input text.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the computed AX-G metrics.
    """
    # Load the AX-G dataset
    dataset = dl.load_dataset("axg")  # Instantiate the AX-G dataset loader
    subset = dataset[:subset_length]

    predictions = []  # List to store model predictions (as integers)
    references = []  # List to store reference labels (as integers)

    # Loop over each entry in the dataset subset
    for entry in subset:
        premise = entry['premise']
        hypothesis = entry['hypothesis']
        label = entry['label']  # 0 = "not entailment", 1 = "entailment"

        # Create the input text by combining the premise and hypothesis.
        # Note: The expected response is 1 for "not entailment" and 0 for "entailment" (as per instruction).
        input_text = (
            f"Sentence 1: {premise}\n"
            f"Sentence 2: {hypothesis}\n"
            "Answer with either 1 for 'not entailment' or 0 for 'entailment'.Do not answer with text!"
        )
        try:
            # Call the model with the input text
            response = model(input_text, max_new_tokens=50)
        except Exception as e:
            # If an error occurs, print model name, error details, and the dataset entry, then skip this entry
            print(model.model_name)
            print(e)
            print(entry)
            continue

        try:
            # Convert the model's response to an integer
            prediction_label = int(response.strip())
        except Exception as e:
            # If the response cannot be converted to an integer, log an error and skip the entry
            print(model.model_name)
            print(response)
            print("Incorrect response format - skipping entry")
            print(e)
            continue

        # Append the prediction and reference label to their respective lists
        predictions.append(prediction_label)
        references.append(label)

    # Prepare output data with predictions and references
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a dynamic directory for storing the model-specific output
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Set the output filename
    output_file = os.path.join(output_dir, "axg_predictions_references.json")

    # Write the output data to a JSON file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the AX-G metric using the predictions and references
    results = axg_metric(predictions, references)

    return results


def axg_eval(models, subset_length=10):
    """
    Evaluates the AX-G task for all provided models and saves the aggregated results to a JSON file.

    AX-G is an extension to the core SuperGLUE metrics.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for each model's evaluation.

    Returns:
        dict: A dictionary containing the evaluation results for all models.
    """
    all_results = {}
    for model in models:
        result = compute_axg(model, subset_length)
        all_results[model.model_name] = result

    # Create a directory for final aggregated results
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)

    # Set the output filename for the aggregated results
    results_file = os.path.join(results_dir, "axg_all_models_results.json")

    # Write the aggregated results to a JSON file
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All AX-G results have been saved to {results_file}.")

    return all_results
