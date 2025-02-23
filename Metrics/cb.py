import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


# Define the CB metric calculation function.
def cb_metric(predictions, references):
    """
    Computes the CB metric (Matthews Correlation Coefficient and Accuracy) for the given predictions.

    This function uses the `evaluate` library to calculate the CB metrics based on the predictions and reference labels.

    Args:
        predictions (list): A list of predictions as integers.
        references (list): A list of reference labels as integers.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    metric = load("super_glue", "cb")
    results = metric.compute(predictions=predictions, references=references)
    return results


# Extended function to evaluate a model on the CB dataset.
def compute_cb(model, subset_length=10):
    """
    Evaluates a model on a subset of the CB dataset from SuperGLUE and saves the results to a file.

    This function loads a subset of the CB dataset, uses the model to generate predictions, and saves both the predictions
    and the reference answers in a JSON file. Finally, it computes the CB metrics using the predictions and references.

    Args:
        model (function): A function that calls the model and returns an answer for the given text.
        subset_length (int): The number of entries from the CB dataset to use for evaluation.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Load the CB dataset.
    dataset = dl.load_dataset("cb")  # Instantiate the CB dataset loader.
    subset = dataset[:subset_length]

    # Lists to store predictions and reference labels.
    predictions = []
    references = []

    # Loop over the subset of the dataset.
    for entry in subset:
        premise = entry['premise']
        hypothesis = entry['hypothesis']
        label = entry['label']  # 0 = "entailment", 1 = "neutral", 2 = "contradiction"

        # Create the input text for the model.
        input_text = (
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            "Determine if the hypothesis is entailment (0), contradiction (1), or neutral (2) based on the premise. "
            "Respond only with 0, 1, or 2. Do not answer with text!"
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

        # Append the prediction as an integer.
        predictions.append(prediction_label)
        # Append the reference label.
        references.append(label)

    # Save the predictions and references in a JSON file.
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a dynamic directory to save the results in a model-specific folder.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename.
    output_file = os.path.join(output_dir, "cb_predictions_references.json")

    # Write the output data to a JSON file.
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the CB metrics using the predictions and references.
    results = cb_metric(predictions, references)

    return results


# Function to evaluate the CB metric for all models.
def cb_eval(models, subset_length=10):
    """
    Evaluates the CB metric for all provided models and saves the aggregated results in a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the evaluation results for all models.
    """
    all_results = {}
    for model in models:
        result = compute_cb(model, subset_length)
        all_results[model.model_name] = result

    # Create a directory for final results.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "cb_all_models_results.json")

    # Write the aggregated results to a JSON file.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All CB results have been saved to {results_file}.")

    return all_results
