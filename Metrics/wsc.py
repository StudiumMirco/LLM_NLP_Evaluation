import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


# Define the WSC metric
def wsc_metric(predictions, references):
    """
    Computes the WSC metric (accuracy) for the given predictions.

    This function uses the `evaluate` library to calculate the accuracy of the predictions.

    Args:
        predictions (list): A list of predictions as integers.
        references (list): A list of reference labels as integers.

    Returns:
        dict: A dictionary containing the computed metric, including accuracy.
    """
    metric = load("super_glue", "wsc")
    results = metric.compute(predictions=predictions, references=references)
    return results


# Extended compute_wsc method
def compute_wsc(model, subset_length=10):
    """
    Evaluates a model on a subset of the WSC dataset from SuperGLUE and saves the results to a file.

    This function loads a subset of the WSC dataset, uses the provided model to generate predictions,
    and saves both the predictions and reference answers.

    Args:
        model (function): A function that calls the model and returns an answer for the given input text.
        subset_length (int): The number of samples from the WSC dataset to use for evaluation.

    Returns:
        dict: A dictionary containing the computed metrics.
    """

    # Load the WSC dataset
    dataset = dl.load_dataset("wsc")  # Instantiate the WSC dataset loader
    subset = dataset[:subset_length]

    # Lists to store the results
    predictions = []
    references = []

    # Loop over the subset of the dataset
    for entry in subset:
        sentence = entry['text']
        span1_text = entry['span1_text']
        span2_text = entry['span2_text']
        label = entry['label']  # 0 = "not coreferent", 1 = "coreferent"

        # Create the input text for the model
        input_text = (
            f"Sentence: {sentence}\n"
            f"Does '{span1_text}' refer to '{span2_text}'? Answer with either 0 for 'no' or 1 for 'yes'. Do not answer with text!"
        )

        try:
            # Call the model with the input text
            response = model(input_text, max_new_tokens=50)

        # Catch model errors; if an error occurs, skip this data entry
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        try:
            # Convert the model's response to an integer value
            prediction_label = int(response.strip())

        # If the model's response is not in the correct format, skip the data entry
        except Exception as e:
            print(model.model_name)
            print(response)
            print("Incorrect response format - skipping entry")
            print(e)
            continue

        # Store the prediction (as an integer) in the predictions list
        predictions.append(prediction_label)

        # Store the reference label (as an integer) in the references list
        references.append(label)

    # Save the predictions and references to a JSON file
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Dynamic path to save the results in a folder specific to the model
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename
    output_file = os.path.join(output_dir, "wsc_predictions_references.json")

    # Write the data to a JSON file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the WSC metrics
    results = wsc_metric(predictions, references)

    return results


def wsc_eval(models, subset_length=10):
    """
    Performs the WSC evaluation for all provided models and saves the results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of samples to use from the WSC dataset for each model.
    """
    all_results = {}
    for model in models:
        result = compute_wsc(model, subset_length)
        all_results[model.model_name] = result

    # Create a folder for the final results and save them in a JSON file
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "wsc_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All WSC results have been saved to {results_file}.")

    return all_results
