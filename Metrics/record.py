import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


# Define the ReCoRD metric calculation function.
def record_metric(predictions, references):
    """
    Computes the ReCoRD metric (F1 and Exact Match) for the predictions.

    This function uses the `evaluate` library to calculate the F1 and Exact Match scores for the predictions.

    Args:
        predictions (list): A list of prediction dictionaries, where each dictionary contains 'idx' and 'prediction'.
        references (list): A list of reference dictionaries, where each dictionary contains 'idx' and 'answers'.

    Returns:
        dict: A dictionary containing the computed metrics, including F1 and Exact Match.
    """
    # Load the ReCoRD metric from the evaluate library (using the SuperGLUE record task).
    metric = load("super_glue", "record")
    # Format the predictions to the expected structure.
    formatted_predictions = [
        {'idx': pred['idx'], 'prediction_text': pred['prediction']} for pred in predictions
    ]
    # Format the references to the expected structure.
    formatted_references = [
        {'idx': ref['idx'], 'answers': ref['answers']} for ref in references
    ]
    # Compute the metric using the formatted predictions and references.
    results = metric.compute(predictions=formatted_predictions, references=formatted_references)
    return results


# Extended compute_record method.
def compute_record(model, subset_length=10):
    """
    Evaluates a model on a subset of the ReCoRD dataset from SuperGLUE and saves the results to a file.

    This function loads a subset of the ReCoRD dataset, uses the model to generate predictions,
    and saves both the predictions and reference answers.

    Args:
        model (function): A function that calls the model and returns an answer for the given input text.
        subset_length (int): The number of entries from the ReCoRD dataset to use for evaluation.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Load the ReCoRD dataset.
    dataset = dl.load_dataset("record")  # Instantiate the ReCoRD dataset loader.
    subset = dataset[:subset_length]

    # Initialize lists to store the predictions and reference answers.
    predictions = []
    references = []

    # Loop through each entry in the subset.
    for entry in subset:
        passage = entry['passage']  # The passage from which to extract the answer.
        query = entry['query']  # The query for which the answer is required.
        answers = entry['answers']  # The list of reference answers.
        idx = entry['idx']  # The unique identifier of the entry.

        # Create the input text for the model.
        input_text = (
            f"Passage: {passage}\n"
            f"Query: {query}\n"
            f"Provide the best answer from the passage. Your answer should just contain the text for the placeholder and nothing else!"
        )
        try:
            # Call the model with the constructed input text and request up to 50 new tokens.
            response = model(input_text, max_new_tokens=50)
        except Exception as e:
            # If an error occurs, print the model's name, error details, and the entry, then skip to the next entry.
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Clean the model's response by stripping whitespace.
        prediction_text = response.strip()

        # Append the prediction in the expected dictionary format.
        predictions.append({
            'idx': idx,
            'prediction': prediction_text
        })

        # Append the reference answers in the expected dictionary format.
        references.append({
            'idx': idx,
            'answers': answers
        })

    # Prepare the output data with predictions and references.
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a dynamic directory path for saving results specific to the model.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename.
    output_file = os.path.join(output_dir, "record_predictions_references.json")

    # Write the output data to a JSON file.
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the ReCoRD metrics using the predictions and references.
    results = record_metric(predictions, references)

    return results


def record_eval(models, subset_length=10):
    """
    Performs ReCoRD evaluation for all provided models and saves the results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of entries from the ReCoRD dataset to use for each model.
    """
    all_results = {}
    # Evaluate each model and store the results.
    for model in models:
        result = compute_record(model, subset_length)
        all_results[model.model_name] = result

    # Determine the current directory and create a directory for final results.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    # Define the output filename for the overall results.
    results_file = os.path.join(results_dir, "record_all_models_results.json")

    # Write all results to a JSON file.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All ReCoRD results have been saved to {results_file}.")

    return all_results
