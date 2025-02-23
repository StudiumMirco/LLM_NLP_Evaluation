import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


# Define the RTE metric
def rte_metric(predictions, references):
    """
    Computes the RTE metric (accuracy) for the predictions.

    This function uses the `evaluate` library to calculate the accuracy of the predictions.

    Args:
        predictions (list): A list of predictions as integers.
        references (list): A list of reference labels as integers.

    Returns:
        dict: A dictionary containing the computed metric, including accuracy.
    """
    metric = load("super_glue", "rte")
    results = metric.compute(predictions=predictions, references=references)
    return results


# Extended compute_rte method
def compute_rte(model, subset_length=10):
    """
    Evaluates a model on a subset of the RTE dataset from SuperGLUE and saves the results to a file.

    This function loads a subset of the RTE dataset, uses the model to generate predictions, and saves
    both the predictions and reference answers.

    Args:
        model (function): A function that calls the model and returns an answer for the given text.
        subset_length (int): The number of entries from the RTE dataset to use for evaluation.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Load the RTE dataset
    dataset = dl.load_dataset("rte")  # Instantiate the RTE dataset loader
    subset = dataset[:subset_length]

    # Lists to store the results
    predictions = []
    references = []

    # Loop over the subset of the dataset
    for entry in subset:
        premise = entry['premise']
        hypothesis = entry['hypothesis']
        label = entry['label']  # 0 = "not entailment", 1 = "entailment"

        # Create the input text for the model
        input_text = (
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Is the hypothesis entailed by the premise? Answer with either 0 for 'yes' or 1 for 'no'. Do not answer with text!"
        )

        try:
            # Call the model with the input text
            response = model(input_text, max_new_tokens=50)
        except Exception as e:
            # Catch model errors: if an error occurs, skip this data entry
            print(model.model_name)
            print(e)
            print(entry)
            continue

        try:
            # Convert the response to an integer value
            prediction_label = int(response.strip())
        except Exception as e:
            # If the model's response is not in the correct format, skip this data entry
            print(model.model_name)
            print(response)
            print("Incorrect response format - skipping entry")
            print(e)
            continue

        # Append the prediction (as an integer) to the predictions list
        predictions.append(prediction_label)

        # Append the reference label (as an integer) to the references list
        references.append(label)

    # Save the predictions and references to a JSON file
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a dynamic path for saving the results in a model-specific folder
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename
    output_file = os.path.join(output_dir, "rte_predictions_references.json")

    # Write the output data to a JSON file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the RTE metric using the collected predictions and references
    results = rte_metric(predictions, references)

    return results


def rte_eval(models, subset_length=10):
    """
    Performs RTE evaluation for all provided models and saves the results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of entries from the RTE dataset to use for each model.
    """
    all_results = {}
    for model in models:
        result = compute_rte(model, subset_length)
        all_results[model.model_name] = result

    # Create a folder for the final results and save them in a JSON file
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "rte_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All RTE results have been saved to {results_file}.")

    return all_results
