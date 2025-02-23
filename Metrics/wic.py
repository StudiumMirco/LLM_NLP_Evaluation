import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl

# Define the WiC metric (accuracy) calculation function.
def wic_metric(predictions, references):
    """
    Computes the WiC metric (accuracy) for the given predictions.

    This function uses the `evaluate` library to calculate the accuracy of the predictions.

    Args:
        predictions (list): A list of predictions as integers.
        references (list): A list of reference labels as integers.

    Returns:
        dict: A dictionary containing the computed metric, including accuracy.
    """
    metric = load("super_glue", "wic")
    results = metric.compute(predictions=predictions, references=references)
    return results

# Extended compute_wic method.
def compute_wic(model, subset_length=10):
    """
    Evaluates a model on a subset of the WiC dataset from SuperGLUE and saves the results to a file.

    This function loads a subset of the WiC dataset, uses the model to generate predictions,
    and saves both the predictions and the reference answers.

    Args:
        model (function): A function that calls the model and returns an answer for the given input text.
        subset_length (int): The number of data samples from the WiC dataset to use for evaluation.

    Returns:
        dict: A dictionary containing the computed metrics.
    """

    # Load the WiC dataset
    dataset = dl.load_dataset("wic")  # Instantiate the WiC dataset loader
    subset = dataset[:subset_length]

    # Lists to store predictions and reference labels
    predictions = []
    references = []

    # Iterate over the subset of the dataset
    for entry in subset:
        sentence1 = entry['sentence1']
        sentence2 = entry['sentence2']
        word = entry['word']
        label = entry['label']  # 0 = "different meaning", 1 = "same meaning"

        # Construct the input text for the model
        input_text = (
            f"Word: {word}\n"
            f"Sentence 1: {sentence1}\n"
            f"Sentence 2: {sentence2}\n"
            f"Does the word have the same meaning in both sentences? Answer with either 0 for 'no' or 1 for 'yes'. Do not answer with text!"
        )

        try:
            # Call the model with the constructed input text
            response = model(input_text, max_new_tokens=50)

        except Exception as e:
            # If an error occurs during model prediction, print the error and skip the entry.
            print(model.model_name)
            print(e)
            print(entry)
            continue

        try:
            # Convert the model's response to an integer value
            prediction_label = int(response.strip())

        except Exception as e:
            # If the model's response is not in the correct format, print an error and skip the entry.
            print(model.model_name)
            print(response)
            print("Incorrect response format - skipping entry")
            print(e)
            continue

        # Append the prediction (as an integer) to the predictions list
        predictions.append(prediction_label)

        # Append the reference label (as an integer) to the references list
        references.append(label)

    # Prepare the output data containing predictions and references.
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Construct a dynamic directory path for saving results specific to the model.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename.
    output_file = os.path.join(output_dir, "wic_predictions_references.json")

    # Write the predictions and references to the JSON file.
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the WiC metric using the collected predictions and references.
    results = wic_metric(predictions, references)

    return results

def wic_eval(models, subset_length=10):
    """
    Performs the WiC evaluation for all provided models and saves the results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of data samples from the WiC dataset to use for each model.
    """
    all_results = {}
    for model in models:
        result = compute_wic(model, subset_length)
        all_results[model.model_name] = result

    # Create a directory for the final results and save them in a JSON file.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "wic_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All WiC results have been saved to {results_file}.")

    return all_results
